from typing import List
import io
import zipfile
from PIL import Image
import logging

from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
import uvicorn

from ml import predict_digit

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
_logger.addHandler(stream_handler)

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Prediction for a single file
    """
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((8, 8)).convert("L")
    except Exception as err:
        _logger.error("Could not convert image %s: %s", file.filename, repr(err))
        return JSONResponse(content={"msg": "could not convert input file to image"},
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    digit, confidence = predict_digit(image)
    _logger.info("Inference: file name {:s}, confidence {:4.3f}, prediction {:d}".format(file.filename, confidence, digit))
    return JSONResponse(content={"digit": digit, "confidence": confidence, "filename": file.filename})


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Batch prediction for multiple files
    """
    predictions = []
    for file in files:
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(await file.read())) as archive:
                for filename in archive.namelist():
                    with archive.open(filename) as image_file:
                        try:
                            image = Image.open(image_file)
                            image = image.resize((8, 8)).convert("L")
                        except Exception as err:
                            _logger.error("Could not convert image %s: %s", filename, repr(err))
                            continue
                        digit, confidence = predict_digit(image)
                        _logger.info("Inference: file name {:s}, confidence {:4.3f}, prediction {:d}"
                                     .format(filename, confidence, digit))
                        predictions.append({"filename": filename,
                                            "digit": digit,
                                            "confidence": confidence})
        else:
            try:
                image = Image.open(io.BytesIO(await file.read()))
                image = image.resize((8, 8)).convert("L")
            except Exception as err:
                _logger.error("Could not convert image %s: %s", file.filename, repr(err))
                continue
            digit, confidence = predict_digit(image)
            _logger.info("Inference: file name {:s}, confidence {:4.3f}, prediction {:d}"
                         .format(file.filename, confidence, digit))
            predictions.append({"filename": file.filename,
                                "digit": digit,
                                "confidence": confidence})
    return JSONResponse(content=predictions)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)

