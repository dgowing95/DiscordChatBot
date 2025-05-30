import os
import torch
import io
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

class ImageHandler:
    pipe: StableDiffusionImg2ImgPipeline = None

    def __init__(self):
        pass

    async def setup(self):
        if ImageHandler.pipe is None:
            print("Initializing Img2Img pipeline...")
            ImageHandler.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-2-1"),
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir="/home/.cache/huggingface/hub",
            ).to("cuda")
            ImageHandler.pipe.safety_checker = None
            print("Img2Img pipeline ready.")

    async def image_to_image(self, prompt: str, init_image: Image.Image, strength: float = 0.75):
        await self.setup()
        try:
            output = ImageHandler.pipe(prompt=prompt, image=init_image, strength=strength)
            return self.return_image_bytes(output.images[0])
        except Exception as e:
            print(f"Error during image-to-image generation: {e}")
            return None

    def return_image_bytes(self, image: Image.Image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array