FROM alpine
FROM python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 vim htop  -y 
RUN mkdir /source
COPY . /source
WORKDIR /source
RUN pip install -r requirements.txt
EXPOSE  8000
