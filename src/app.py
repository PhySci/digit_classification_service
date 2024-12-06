from typing import List
import io
import zipfile
from PIL import Image
import logging

from fastapi import FastAPI, File, UploadFile
import uvicorn

from ml import predict_digit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction for a single file
    """
    image = Image.open(io.BytesIO(await file.read()))
    digit, confidence = predict_digit(image)
    return {"digit": int(digit), "confidence": float(confidence)}

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple files
    """
    predictions = []
    for file in files:
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(await file.read())) as archive:
                for filename in archive.namelist():
                    with archive.open(filename) as image_file:
                        image = Image.open(image_file)
                        digit, confidence = predict_digit(image)
                        predictions.append({"digit": digit, "confidence": confidence})
        else:
            image = Image.open(io.BytesIO(await file.read()))
            digit, confidence = predict_digit(image)
            predictions.append({"digit": digit, "confidence": confidence})
    
    return predictions

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

