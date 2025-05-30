from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.handler import ImageHandler
from PIL import Image
import io

app = FastAPI()
handler = ImageHandler()

@app.post("/img2img")
async def img2img(
    prompt: str = Form(...),
    strength: float = Form(0.75),
    file: UploadFile = File(...)
):
    init_bytes = await file.read()
    try:
        init_image = Image.open(io.BytesIO(init_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    output_bytes = await handler.image_to_image(prompt, init_image, strength)
    if output_bytes is None:
        return {"error": "Image generation failed"}

    return StreamingResponse(output_bytes, media_type="image/png")