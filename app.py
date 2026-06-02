from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
from rembg import remove
import numpy as np
import cv2

app = FastAPI()

# ✅ 解決 CORS（你 HTML fetch 失敗主因之一）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    url: str


def load_image(url):
    r = requests.get(url, timeout=10)
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    return img


@app.get("/")
def home():
    return {"status": "ok", "msg": "API running"}

@app.post("/predict")
def predict(req: Req):
    img = load_image(req.url)

    # remove bg
    out = remove(img)

    if isinstance(out, np.ndarray):
        out = Image.fromarray(out)

    out = out.convert("RGBA")

    # simple metrics
    arr = np.array(out)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 0)

    if len(xs) == 0:
        return {
            "status": "error",
            "msg": "empty mask"
        }

    shoulder_ratio = (np.max(xs) - np.min(xs)) / arr.shape[1]
    offset_ratio = np.min(ys) / arr.shape[0]

    return {
        "status": "OK",
        "shoulder_ratio": float(shoulder_ratio),
        "offset_ratio": float(offset_ratio)
    }
