import cv2
import os, datetime
from typing import Optional
from pydantic import BaseModel
from dotenv import dotenv_values
from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List



# # env
# if os.environ.get("ENV") == None:
#     config = dotenv_values(".env")
#     # camera_id = config["CAMERA_ID"]
#     camera_url = config["CAMERA_URL"]
#     fps = config["FPS"]
#     scale = config["SCALE"]

# else:
#     # camera_id = os.environ.get("CAMERA_ID")
#     camera_url = os.environ.get("CAMERA_URL")
#     fps = os.environ.get("FPS")
#     scale = os.environ.get("SCALE")

router = APIRouter()

# request Body
class Camera(BaseModel):
    camera_id: int
    camera_url: str
    fps: int
    scale: str
    # description: Optional[str] = None


@router.post("/camera")
async def makeImage(camera: Camera):
    filepath = camera.camera_url
    print("camera",camera)
    fps = camera.fps
    video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함
    print(os.path.exists(filepath[:-4]))
    # while(video.isOpened()):
    #     ret, image = video.read()
    #     if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
    #         cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
    #         print('Saved frame number :', str(int(video.get(1))))
    #         count += 1
    # video.release()
    return 0
    
    

# if not video.isOpened():
#     print("Could not Open :", filepath)
#     exit(0)


# #불러온 비디오 파일의 정보 출력
# length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = video.get(cv2.CAP_PROP_FPS)

# print("length :", length)
# print("width :", width)
# print("height :", height)
# print("fps :", fps)    

# #프레임을 저장할 디렉토리를 생성
# try:
#     if not os.path.exists(filepath[:-4]):
#         os.makedirs(filepath[:-4])
# except OSError:
#     print ('Error: Creating directory. ' +  filepath[:-4])


# count = 0

# while(video.isOpened()):
#     ret, image = video.read()
#     if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
#         cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
#         print('Saved frame number :', str(int(video.get(1))))
#         count += 1
        
# video.release()




def writeVideo():
    #현재시간 가져오기
    currentTime = datetime.datetime.now()
    
    #RTSP를 불러오는 곳
    video_capture = cv2.VideoCapture('rtsp://admin:admin@192.168.0.2:554')
    
    # 웹캠 설정
    video_capture.set(3, 800)  # 영상 가로길이 설정
    video_capture.set(4, 600)  # 영상 세로길이 설정
    fps = 20
    # 가로 길이 가져오기
    streaming_window_width = int(video_capture.get(3))
    # 세로 길이 가져오기
    streaming_window_height = int(video_capture.get(4))  
    
    #현재 시간을 '년도 달 일 시간 분 초'로 가져와서 문자열로 생성
    fileName = str(currentTime.strftime('%Y %m %d %H %M %S'))

    #파일 저장하기 위한 변수 선언
    path = f'D:/cctv/cctv/python/{fileName}.avi'
    
    # DIVX 코덱 적용 # 코덱 종류 # DIVX, XVID, MJPG, X264, WMV1, WMV2
    # 무료 라이선스의 이점이 있는 XVID를 사용
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    
    # 비디오 저장
    # cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
    out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))

    while True:
        ret, frame = video_capture.read()
        # 촬영되는 영상보여준다. 프로그램 상태바 이름은 'streaming video' 로 뜬다.
        cv2.imshow('streaming video', frame)
        
        # 영상을 저장한다.
        out.write(frame)
        
        # 1ms뒤에 뒤에 코드 실행해준다.
        k = cv2.waitKey(1) & 0xff
        #키보드 esc 누르면 종료된다.
        if k == 27:
            break
    video_capture.release()  # cap 객체 해제
    out.release()  # out 객체 해제
    cv2.destroyAllWindows()