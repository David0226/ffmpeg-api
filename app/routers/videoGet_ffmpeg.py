import ffmpeg
import os

print(os.getcwd())
path = os.getcwd()
# ffmpeg cmd
# ffmpeg -y -rtsp_transport tcp -c:v hevc_nvv4l2dec -i rtsp://admin:!q1w2e3r4@192.168.1.105:554 -vf fps=1 -strftime 1 /home/bnd/skcc/image/output/cam105_%Y%m%d_%H%M%S.jpg


# ffmpeg.input('rtsp://admin:#skcc09628@192.168.50.127:554',ss=1).filter('fps', fps=25, round='up').output('D:\10.Source\13. Becom\Image_API\app\routers\dummy2.mp4').run()

stream = ffmpeg.input(path+'\\avatar.mp4',ss=1) 
# stream = ffmpeg.input('rtsp://admin:#skcc09628@192.168.50.127:554')
print(stream)
# capture = ffmpeg_streaming.input('CAMERA NAME OR SCREEN NAME', capture=True)

# stream = ffmpeg.input('dummy.png')
# stream = ffmpeg.filter(stream, 'fps', fps=25, round='up')
stream = ffmpeg.output(stream, path+'\\test2.png')
print(stream)
ffmpeg.run(stream)
print(stream)