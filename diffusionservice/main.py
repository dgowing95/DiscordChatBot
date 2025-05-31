from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from text_to_image_handler import TextToImageHandler
from image_to_image_handler import ImageToImageHandler
import io

app = FastAPI()
text_to_image_handler = TextToImageHandler()
image_to_image_handler = ImageToImageHandler()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/text-image")
async def textToImage(prompt: PromptRequest):
    image_bytes = await text_to_image_handler.generate_image(prompt.prompt)
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Image generation failed")
    
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/image-image")
async def imageToImage(
    prompt: str = Form(...),
    strength: float = Form(0.75),
    file: UploadFile = File(...)
):
    init_bytes = await file.read()
    try:
        init_image = Image.open(io.BytesIO(init_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    output_bytes = await image_to_image_handler.image_to_image(prompt, init_image)
    if output_bytes is None:
        return {"error": "Image generation failed"}

    return StreamingResponse(output_bytes, media_type="image/png")