import base64
from io import BytesIO
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from matplotlib import pyplot as plt
from predict import predict_image
from PIL import Image

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_data = await file.read()
    original_image = Image.open(BytesIO(image_data)).convert("RGB")
    prediction, predicted_prob, heat_map = predict_image(original_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    result = {
        "classification": "Positive" if prediction == 1 else "Negative",
        "confidence": predicted_prob if prediction == 1 else (1 - predicted_prob),
        "image": base64.b64encode(heat_map).decode('utf-8')
    }

    return JSONResponse(content=result)
