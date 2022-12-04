### MSC Carriage 내 Tray 위치 인식 프로그램 구조 설명
```
├── config
│   └── config.json            : configuration file
│
├── detect_model
│   ├── 20221018_realdata3_e2750_best.pth    : mmpose로 학습시킨 checkpoint file
│   └── mmpose_msc_higherhrnet_w48_coco_512x512_udp.py   : mmpose config file
│
├── source_data
│   ├── carriage_tray_1.mp4    : detect용으로 사용할 source video file (tray 1단)
│   └── carriage_tray_2.mp4    : detect용으로 사용할 source video file (tray 1단)
│
├── switch
│   └── detect_switch.json     : detection 기능 On/Off 제어를 위한 파일
│
├── result                     : 결과 파일 저장 폴더
│   ├── camera001
│   ├── camera001
│   └── recent
│
├── msc_detect.py              : main program
├── msc_config.py              : configuration 관리를 위한 program
├── pose_calc.py               : 좌표변환 및 위치계산을 위한 program
│
└── Readme.md                  : 설명 파일
```

### 2 request body example

{
   "dvce_id" : "cam01",  		         => 연결 카메라의 ID     
   "strm_url" : "rtsp://admin:!q1w2e3r4@192.168.1.111:554", => 카메라 접속정보 
   "fps" : “1", 			         => 자를 프레임 수
   "out_dir" : "/home/spring/data/output"  	         => 컨테이너 내 이미지 떨어지는 경로(docker mount를 통해 DAD서버 내 폴더와 연동되어 있음)
   “scale” : “800X480”, 	  	         => 이미지 사이즈, default = “”
   “crop” : “1000X1000” 		         => crop 할 영역, default = “”
}

CAMERA_ID = camera01
CAMERA_URL = "D:\10.Source\13. Becom\Image_API\app\routers\avatar.mp4"
FPS = 1
SCALE = ""

### 3 running cmd
uvicorn main:app 0.0.0.0:8000 --reload
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600