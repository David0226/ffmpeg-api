import os.path
import ffmpeg, ffmpeg_streaming

# ffmpeg cmd
# ffmpeg -y -rtsp_transport tcp -c:v hevc_nvv4l2dec -i rtsp://admin:!q1w2e3r4@192.168.1.105:554 -vf fps=1 -strftime 1 /home/bnd/skcc/image/output/cam105_%Y%m%d_%H%M%S.jpg



mp4_input = ffmpeg.input('./acatar.mp4') 
video = ffmpeg_streaming.input('rtsp://hostname[:port]/path')
capture = ffmpeg_streaming.input('CAMERA NAME OR SCREEN NAME', capture=True)
