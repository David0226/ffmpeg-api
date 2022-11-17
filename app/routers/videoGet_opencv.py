import cv2
import os
from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

app = FastAPI()

# env
if os.environ.get("ENV") == None:
    config = dotenv_values(".env")
    CAMERA_ID = config["CAMERA_ID"]
    CAMERA_URL = config["CAMERA_URL"]
    FPS = config["FPS"]
    SCALE = config["SCALE"]

else:
    CAMERA_ID = os.environ.get("CAMERA_ID")
    CAMERA_URL = os.environ.get("CAMERA_URL")
    FPS = os.environ.get("FPS")
    SCALE = os.environ.get("SCALE")

@app.post(mkImage/{CAMERA_ID})
async def makeImage()


filepath = CAMERA_URL
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함
print(os.path.exists(filepath[:-4]))

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)


#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)    

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