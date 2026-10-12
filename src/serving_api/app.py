import sys
from io import BytesIO

import uvicorn
import numpy as np
import tensorflow as tf

from PIL import Image
from loguru import logger
from fastapi import FastAPI, Response, status, File, UploadFile

from src.base.model import CNNModel

# Set up logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add(sys.stdout, filter=lambda record: record["level"].no < 40, level="INFO")


# Create the App
app = FastAPI(title="Meat Quality Assessment")

# Initialise global parameters to be filled at startup
global_items = {}


# Run at startup - load the model from a local path
@app.on_event("startup")
def startup() -> None:
    """Load and initialise the model."""
    path = "model"
    logger.info(f"Loading model files from {path}.")
    global_items["model"] = CNNModel.load_model(path)
    logger.info("Model loaded successfully.")


@app.get("/health")
async def health_check() -> None:
    """Endpoint for health check."""
    if "model" in global_items and global_items["model"] is not None:
        return Response("healthy", status_code=status.HTTP_200_OK)
    else:
        return Response("not healthy", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.post("/predict")
async def prediction(image: UploadFile = File(...)) -> dict:
    """Make a prediction on an image using the model.

    Args:
        image (File): Uploaded image file

    Returns:
        dict: API response
    """
    try:
        logger.info(f"Received image {image.filename}")
        image_bytes = await image.read()
        image = np.array(Image.open(BytesIO(image_bytes)))
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (256, 256))
        image = tf.expand_dims(image, axis=0)
        preds_proba = global_items["model"].predict(image)
        response = {
            "pred_fresh": str(preds_proba[0][0]),
            "pred_spoiled": str(preds_proba[0][1]),
            "pred_class": str(np.argmax(preds_proba[0])),
        }
    except Exception as error:
        response = {"error": f"{error}"}
    finally:
        return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False, log_level="info")
