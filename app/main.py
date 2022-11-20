from threading import Thread
from fastapi import FastAPI
from dotenv import dotenv_values
from routers import videoGet_opencv
import uvicorn




app = FastAPI()
app.include_router(videoGet_opencv.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}



