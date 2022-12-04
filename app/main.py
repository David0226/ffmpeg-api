from threading import Thread
from fastapi import FastAPI
from dotenv import dotenv_values
from routers import msc_detect
import uvicorn


app = FastAPI()
app.include_router(msc_detect.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}



