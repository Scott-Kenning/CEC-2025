from fastapi import FastAPI, File, UploadFile
#import base64
from predict import *

app = FastAPI()

@app.post("/classify")
async def classify(image_file: UploadFile = File(...)):

    if (image_file.content_type != "image/jpeg" and image_file.content_type != "image/png"):
        return {
            "error": "Invalid file type"
        }

    prediction, predicted_prob = predict_image(image_file.filename)
    
    # Convert the processed image to a Base64-encoded string
    # buffered = io.BytesIO()
    # processed_image.save(buffered, format="PNG")
    # processed_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "classification": "Positive" if prediction == 1 else "Negative",
        "confidence": predicted_prob
    }
 
    
