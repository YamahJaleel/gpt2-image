from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io, os

# Cache settings for Hugging Face
os.environ["HF_HOME"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache/hub"

app = FastAPI()

# Load the ViT-GPT2 captioning model
pipe = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=-1  # -1 = CPU only (good for Render Free plan)
)

@app.get("/")
def home():
    return {"message": "API is running. Use POST /predict with an image."}

@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Generate caption
    result = pipe(image, max_new_tokens=32)
    caption = result[0]["generated_text"].strip()

    return JSONResponse({"caption": caption})
