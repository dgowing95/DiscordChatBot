from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.image_handler import ImageHandler
import io

app = FastAPI()
image_handler = ImageHandler()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(prompt: PromptRequest):
    image_bytes = await image_handler.generate_image(prompt.prompt)
    if image_bytes is None:
        raise HTTPException(status_code=500, detail="Image generation failed")
    
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")