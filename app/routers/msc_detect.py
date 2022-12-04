from mmpose.apis import (inference_bottom_up_pose_model,
                         init_pose_model, vis_pose_result)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import os.path as osp
import sys
import time

import argparse
import json
import shutil

from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter

# import my modules
from config import msc_config
from custom_def.pose_calc import transform_and_calculate_deviation

router = APIRouter(
    prefix="/mscdetect",
    tags=["mscdetect"],
    responses={404: {"description": "Not found"}},
)

local_runtime = False
try:
    from google.colab.patches import cv2_imshow  # for image visualization in colab
except:
    local_runtime = True

class Camera(BaseModel):
    camera_id: int
    camera_url: str
    fps: int
    scale: str

# get commandline arguments
def get_argument():

    parser = argparse.ArgumentParser(
        prog='Detect Tray Position v0.1',
        description='Program to Detect Keypoints & Position of Tray')

    parser.add_argument('config', help='config file in json format')
    parser.add_argument(
        'index', type=int, help='index of section to be acquired from config file')

    args = parser.parse_args()

    print('Detect Tray Position v0.1\n')

    print('[commandline arguments]')
    print('- config:', args.config)
    print('- index:', args.index, '\n')

    return args


# initialize like loading config or model ..
def initialize(cfg_file, index=0):
    time_1 = datetime.now()

    # load configs
    config = msc_config.load_config(cfg_file)

    if index >= len(config):
        sys.exit('Error: Specified invalid index of config!')

    # index에 해당되는 정보만 취득
    c = config[index]

    # initialize pose model
    if c.gpu_device:
        pose_model = init_pose_model(
            c.model_config, c.checkpoint, device=c.gpu_device)
    else:
        pose_model = init_pose_model(c.model_config, c.checkpoint)

    time_2 = datetime.now()
    print('- initializing time(secs) :', (time_2 - time_1).total_seconds())

    # result_path가 없는 경우, 생성
    try:
        if not os.path.exists(c.result_path):
            os.makedirs(c.result_path)
            print('- create result directory :', c.result_path)
    except OSError:
        sys.exit(f'Error: Failed to create the directory. - {c.result_path}')

    return c, pose_model


# inference image by mmpose - detect 6 keypoints of tray
def inference_image(cfg, model, img, img_fn):
    time_1 = datetime.now()

    # inference pose by bottom-up model
    pose_results, returned_outputs = inference_bottom_up_pose_model(
        model, img, dataset=pose_model.cfg.data.test.type)

    time_2 = datetime.now()  # Inference time : time_2 - time_1

    # json format으로 저장시에 float32 type은 에러 발생 => float type으로 변경
    return {'file_name': osp.basename(img_fn),
            'image_size': img.shape[0:2],
            'keypoints': np.round(pose_results[0]['keypoints'][:, 0:2].astype(np.float), 2).tolist(),
            'scores': np.round(pose_results[0]['keypoints'][:, 2].astype(np.float), 2).tolist(),
            'time_inference': round(float((time_2 - time_1).total_seconds()), 3)
            }


# Shows the duration of the section. when timestamp_ON is True
timestamp_ON = False


def timestamp(tag='', start=False):
    '''
    tag : 위치를 인식하기 위한 tag 정보를 함께 출력함
    start : 처음 시작시 True로 지정하여 호출. t값을 현재시간으로 초기화
    '''
    global t

    if not timestamp_ON:
        return

    if start:
        t = datetime.now()

    t2 = datetime.now()
    print('* [{}] {:.4f} second elapsed ({})'.format(tag,
          (t2 - t).total_seconds(), t2.strftime('%Y-%m-%d %H:%M:%S.%f')))
    t = t2


################# main routine #######################

@router.get("/inference")
def inference_main():

		# get commandline arguments
	# args = get_argument()
	# args = 
	
	# initialize like config & model loading
	print('Initializing...')
	# cfg, pose_model = initialize('/workspace/mydata/AI_Camera/product/config.json', index=0)
	# cfg, pose_model = initialize(args.config, args.index)
	cfg, pose_model = initialize("..\config\config.json", 0)
	
	
	# 카메라 정보 할당. index를 지정하면 카메라, path를 지정하면 동영상 파일
	# IP카메라의 경우, RTSP(Real-time Stream Protocol) URL 지정 => rtsp://id:pw@URL
	print('\nOpenning video...')
	print('- source :', cfg.data_root)
	capture = cv2.VideoCapture(cfg.data_root)
	
	frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
	frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
	frame_fps = capture.get(cv2.CAP_PROP_FPS)
	
	print('\n[video information]')
	print('- width:', frame_width)
	print('- height:', frame_height)
	print('- frames:', frame_count)
	print('- fps:', frame_fps)
	print('- length:', frame_count/frame_fps, 'sec\n')
	
	
	# 저장하는 frame 단위
	# 예를 들어 fps(30) * interval(0.5) => 0.5초 15 frame마다 추출
	extract_fps = int(frame_fps*cfg.detect_interval)
	
	
	# 카메라가 작동중인지 확인. 동영상 파일인 경우 정상 Open여부 확인
	if not capture.isOpened():
		print(f'Cannot open video!!!\n')
		sys.exit()
	
	image_id = 0
	cycle_start = datetime.now()  # frame을 추출하는 시점을 계산하기 위한 변수
	# results = []
	timestamp('start', start=True)
	while True:
		# 카메라에서 frame 읽기  (ret은 상태값(True/False), frame은 이미지)
		ret, img = capture.read()
	
		# 프레임을 읽지 못한 경우, while루프 종료
		if not ret:
			print('                                                    ')
			print('End of video')
			print(f'- Number of images : {image_id}')
			break
	
		# 저장하는 시간단위로 frame 추출. 아닌 경우 skip
		if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) % extract_fps != 0:
			continue
	
		image_id += 1
		cur_time = datetime.now()
		cur_time_s = cur_time.strftime('%Y%m%d_%H%M%S_%f')  # 20220101_120130
		img_fn = os.path.join(cfg.result_path, f'{cur_time_s}.jpg')
	
		timestamp(f'{image_id}:1')
	
		print('Image No [{:04d}] : {}'.format(
			image_id, cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')))
	
		# detect_switch를 읽어서 detect_carriage값이 'ON'이면,
		# detect keypoints -> calc. position -> save results 진행
		with open(cfg.detect_switch, 'r') as f:
			detect_switch = json.load(f)
	
		timestamp(f'{image_id}:2')
	
		if detect_switch['detect_carriage'] == 'ON':
			print(f'- Detection Mode : ON')
	
			result = {
				'camera_id': cfg.camera_id,
				'detect_target': 'Carriage'
			}
	
			# keypoint detection
			print('- Inference-Keypoint Detection...', end='')
			result_inf = inference_image(cfg, pose_model, img, img_fn)
			print('[{} sec]'.format(result_inf['time_inference']))
			result.update(result_inf)  # 결과로그 dict 병합
	
			timestamp(f'{image_id}:3')
	
			# transform coordination & calculation deviations
			print('- Transform Coordi. & Calculating position...')
			result_calc = transform_and_calculate_deviation(
				cfg, img, img_fn, result['keypoints'][0:3])
			result.update(result_calc)  # 결과로그 dict 병합
	
			timestamp(f'{image_id}:4')
	
			# 결과로그에 실행시간 포함
			end_time = datetime.now()
			result.update({
				'log_time_start': cur_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
				'log_time_end': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
				'log_time_total': round(float((end_time - cur_time).total_seconds()), 3)
			})
	
			# result json 파일 저장
			fn, _ = osp.splitext(osp.basename(img_fn))
			result_json = osp.join(cfg.result_path, f'{fn}.json')
			print(f'- Write results : {fn}.json')
			with open(result_json, 'w') as f:
				json.dump(result, f, indent='\t')
	
			timestamp(f'{image_id}:5')
	
			# result_log file에 최신 파일 업데이트
			shutil.copy(result_json, osp.join(cfg.result_path, cfg.result_log))
	
			timestamp(f'{image_id}:6')
	
			print()
			print(f'process time: {(end_time - cur_time).total_seconds():.3f} sec')
	
	#         results.append(result)
	
		# detect_rack값이 'ON'이면, rack에 대한 detection/calculation 진행
	#     if detect_switch['detect_rack'] == 'ON':
			# ...
	
		# switch가 'OFF'면, 원본 이미지 그대로 저장
		else:
			# frame 그대로 저장
			cv2.imwrite(img_fn, img)
			print(f'- Detection Mode : OFF')
			print(f'- Write image : {cur_time_s}.jpg')
			print()
	
			# result_image file에 최신 파일 업데이트
			shutil.copy(img_fn, osp.join(cfg.result_path, cfg.result_image))
	
			timestamp(f'{image_id}:7')
	
		# 처리시간을 감안하여 waiting interval 조절
		cycle_end = datetime.now()
		wait_interval = cfg.detect_interval - \
			(cycle_end - cycle_start).total_seconds()
	
		print('total time: {:.3f} sec ({} - {})'.format((cycle_end - cycle_start).total_seconds(),
														cycle_start.strftime('%Y-%m-%d %H:%M:%S.%f'), cycle_end.strftime('%Y-%m-%d %H:%M:%S.%f')))
		print(f'wait interval: {wait_interval:.3f} sec\n')
	
		if wait_interval > 0:
			time.sleep(wait_interval)
	
		cycle_start = datetime.now()
	
		timestamp(f'{image_id}:8')
	
	# 카메라 장치에서 받아온 메모리 해제
	capture.release()
	a = "sucess"
	return a
