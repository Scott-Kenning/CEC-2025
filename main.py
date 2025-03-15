import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from predict import predict_image

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        return {"error": "Invalid file type"}

    image_data = await file.read()

    print("test", flush=True)

    # Create a temporary file with an appropriate suffix
    suffix = ".jpg" if file.content_type == "image/jpeg" else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name

    try:
        # Call predict_image with the temporary file path
        prediction, predicted_prob, heat_map = predict_image(tmp_path)
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)

    return {
        "classification": "Positive" if prediction == 1 else "Negative",
        "confidence": (1 - predicted_prob) if prediction == 0 else predicted_prob
    }
