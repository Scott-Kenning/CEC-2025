from io import BytesIO
import base64
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from predict import predict_image  # ensure this returns (prediction, predicted_prob, heat_map)

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Validate file type if necessary
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    image_data = await file.read()
    original_image = Image.open(BytesIO(image_data)).convert("RGB")
    prediction, predicted_prob, heat_map = predict_image(original_image)
    
    # Only encode the heatmap image if the classification is positive
    image_field = None
    if prediction == 1:
        # Convert heat_map (a NumPy array) to PNG bytes
        retval, buffer = cv2.imencode('.png', heat_map)
        image_field = base64.b64encode(buffer).decode('utf-8')

    result = {
        "classification": "Positive" if prediction == 1 else "Negative",
        "confidence": predicted_prob if prediction == 1 else (1 - predicted_prob),
        "image": image_field
    }
    return JSONResponse(content=result)
