import cv2
import numpy as np

from datetime import datetime
import os.path as osp
import shutil


# 기준점 정의 - 좌상단, 좌하단, 우하단 3개
def define_base_points(image_size, obj_real_size, length_scale):
    # 이미지 중앙 좌표
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    
    # 좌표변환된 이미지에서 target의 크기 절반 계산
    w_2 = obj_real_size[0] * length_scale / 2
    h_2 = obj_real_size[1] * length_scale / 2

    # 기준점 좌표 3개 계산
    return [[cx-w_2,cy-h_2], [cx-w_2,cy+h_2], [cx+w_2,cy+h_2]]


# 기준점에서 벗어난 x, y 거리를 구하는 함수
def get_deviation_xy(pts, base_pts):
    # pts : 검사 대상 좌표(3개). (x,y)쌍으로된 np.array
    # base_pts : left-bottom 위치의 기준점(1개). (x,y)쌍으로된 tuple
    # return => gap_x, gap_y : 떨어진 x거리, y거리
    
    min_x = min(pts[:,0])
    max_y = max(pts[:,1])
    
    base_x = base_pts[0]
    base_y = base_pts[1]
    
    gap_x = min_x - base_x
    gap_y = max_y - base_y
    
    return(gap_x, gap_y)


# 기준선에서 틀어진 각도를 구하는 함수
def get_rotated_angle(pts, base_pts):
    # pts : 검사 대상 좌표(2개, left-top & left-bottom). (x,y)쌍으로된 np.array
    # base_pts : 기준점(2개, left-top & left-bottom). (x,y)쌍으로된 np.array
    # return => 틀어진 각도(degree). 시계방향은 (+), 반시계방향은 (-)
    
    v = pts[0] - pts[1]
    base_v = base_pts[0] - base_pts[1]
    
    theta = np.degrees(np.arcsin( (base_v[0]*v[1]-base_v[1]*v[0]) / 
                              (np.linalg.norm(base_v) * np.linalg.norm(v)) ))
    
    return(theta)


# 이미지에 정보 출력 : base points, detected points, gap_x, gap_y, theta
def write_result_to_image(cfg, img, img_fn, base_pts, pts, gap_x, gap_y, theta):
    # 기준선을 원본 이미지에 표시
    cv2.polylines(img, [base_pts.astype(np.int32)], False, (0,255,255), 3)

    # 변환된 좌표를 이미지에 표시
    r = 20
    cv2.circle(img, tuple(pts[0]), r, (255,0,0), -1)
    cv2.circle(img, tuple(pts[1]), r, (0,255,0), -1)
    cv2.circle(img, tuple(pts[2]), r, (0,0,255), -1)

    # 변환된 좌표 기준으로 선 표시
    cv2.polylines(img, [pts.astype(np.int32)], False, (255,255,255), 3)

    # 벗어난 거리, 틀어진 각도 정보를 Text로 표시
    fontsize = 2; thick = 5
    cv2.putText(img, f'image : {osp.basename(img_fn)} ({img.shape[1]},{img.shape[0]})', (100,120), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255), thick)
    cv2.putText(img, f'x deviation : {gap_x:+3.1f} mm', (100,190), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255), thick)
    cv2.putText(img, f'y deviation : {gap_y:+3.1f} mm', (100,260), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255), thick)
    cv2.putText(img, f'rotated angle : {theta:+3.1f}', (100,340), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255), thick)

    # 결과 이미지 저장
    cv2.imwrite(img_fn, img)
    print(f'- Write image : {osp.basename(img_fn)}')

    
def transform_and_calculate_deviation(cfg, img, img_fn, pts):
    time_1 = datetime.now()
    
    # Box 하단 기준 꼭지점 좌표 3개
    pts = np.float32(pts)

    # 좌표변환된 이미지의 기준선 지정 ###
    base_pts1 = np.float32(define_base_points(cfg.image_size, cfg.obj_real_size, cfg.length_scale))

    # width, heigh of image
    h, w = img.shape[:2]
    
    # 이미지 원근변환 적용
    img_dst = cv2.warpPerspective(img, cfg.mtrx, (w, h))

    # 좌표 원근변환 적용 - pts.reshape은 차원 추가 목적(2차원 + 1차원 => 3차원)
    pts_dst = cv2.perspectiveTransform(pts.reshape(1,3,2), cfg.mtrx)[0]

    # 벗어난 거리, 틀어진 각도 계산
    gap_x, gap_y = get_deviation_xy(pts_dst, tuple(base_pts1[1]))
    theta = get_rotated_angle(pts_dst, base_pts1)

    # gap_x, gap_y는 scale 비율에 따라 실측값으로 계산하여 표시
    gap_x = gap_x/cfg.length_scale
    gap_y = gap_y/cfg.length_scale
    
    time_2 = datetime.now()
    
    # 원본 이미지에 정보 출력
    write_result_to_image(cfg, img, img_fn, cfg.base_pts0, pts, gap_x, gap_y, theta)
    # 좌표 변환된 이미지에 정보 출력
#     write_result_to_image(cfg, img_dst, img_fn, base_pts1, pts_dst, gap_x, gap_y, theta)

    # result_image file에 최신 파일 업데이트
    shutil.copy(img_fn, osp.join(cfg.result_path, cfg.result_image))


    time_3 = datetime.now()
    
    # json format으로 저장시에 float32 type은 에러 발생 => float type으로 변경
    return {'time_tranf_calc': round(float((time_2 - time_1).total_seconds()), 3), 
            'time_write_img': round(float((time_3 - time_2).total_seconds()), 3),
            'gap_x': round(float(gap_x), 1),
            'gap_y': round(float(gap_y), 1),
            'rotated_angle' : round(float(theta), 1)
    }
