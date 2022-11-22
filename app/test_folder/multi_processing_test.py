import cv2
import os, threading
from typing import Optional
from pydantic import BaseModel
from dotenv import dotenv_values
from fastapi import APIRouter, Body, FastAPI
from fastapi.encoders import jsonable_encoder
from typing import List
# request Body


app = FastAPI()

class Camera(BaseModel):
    camera_id: int
    camera_url: str
    fps: int
    scale: str
    # description: Optional[str] = None

def getCamera(camera):
    # camera_id = camera.camera_id
    # fps = camera.fps
    # scale = camera.scale
    camera_url = camera.camera_url
    video = cv2.VideoCapture(camera_url)
    return video

@app.post("/preCamera")
def makeImage(camera: Camera):
    print(f"{os.getpid()} process | {threading.get_ident()} thread")
    video = getCamera(camera)
    fps = camera.fps
    print(video.get(cv2.CAP_PROP_FPS))
        # video.get() Option
        # cv2.CAP_PROP_FRAME_WIDTH : width 정보
        # cv2.CAP_PROP_FRAME_HEIGHT : height 정보
        # cv2.CAP_PROP_FRAME_COUNT : 영상 총 프레임 수 
        # cv2.CAP_PROP_FPS : 영상 fps값     
    count = 0
    while(video.isOpened()):
        ret, image = video.read() # 영상정보 읽어오기
        if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
            cv2.imwrite("./output" + "/frame%d.jpg" % count, image)
            print('Saved frame number :', str(int(video.get(1))))
            count += 1
        
    video.release()

@app.post("/rawCamera")    
async def makeImage(camera: Camera):
    video = getCamera(camera)
    return 0
