{
   "dvce_id" : "cam01",  		         => 연결 카메라의 ID     
   "strm_url" : "rtsp://admin:!q1w2e3r4@192.168.1.111:554", => 카메라 접속정보 
   "fps" : “1", 			         => 자를 프레임 수
   "out_dir" : "/home/spring/data/output"  	         => 컨테이너 내 이미지 떨어지는 경로(docker mount를 통해 DAD서버 내 폴더와 연동되어 있음)
   “scale” : “800X480”, 	  	         => 이미지 사이즈, default = “”
   “crop” : “1000X1000” 		         => crop 할 영역, default = “”
}

