from threading import Thread
from fastapi import FastAPI
from dotenv import dotenv_values
from routers import videoGet_ffmpeg, videoGet_opencv
import uvicorn




app = FastAPI()