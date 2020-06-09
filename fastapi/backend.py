from fastapi import FastAPI
from typing import List
from fastapi import FastAPI, UploadFile, File
import numpy as np
from starlette.requests import Request
import io
from PIL import Image
import base64
import cv2
app = FastAPI()

@app.post("/predict")
async def analyse(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    return original_image.shape
