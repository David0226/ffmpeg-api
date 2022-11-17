import cv2
import os
from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List



# env
if os.environ.get("ENV") == None:
    config = dotenv_values(".env")
    camera_id = config["CAMERA_ID"]
    camera_url = config["CAMERA_URL"]
    fps = config["FPS"]
    scale = config["SCALE"]

else:
    camera_id = os.environ.get("CAMERA_ID")
    camera_url = os.environ.get("CAMERA_URL")
    fps = os.environ.get("FPS")
    scale = os.environ.get("SCALE")

router = APIRouter()

# request Body
class Camera(BaseModel):
    camera_url: str
    fps: str
    scale: Optional[str] = None

@router.get("/")
async def read_items():
    return fake_items_db


@router.get("/{camera_id}")
async def makeImage(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}


@router.put(
    "/{item_id}",
    tags=["custom"],
    responses={403: {"description": "Operation forbidden"}},
)
async def update_item(item_id: str):
    if item_id != "plumbus":
        raise HTTPException(
            status_code=403, detail="You can only update the item: plumbus"
        )
    return {"item_id": item_id, "name": "The great Plumbus"}

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