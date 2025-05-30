import os, torch, io
from diffusers import DiffusionPipeline

from PIL import Image

class ImageHandler:
    pipe: DiffusionPipeline = None

    def __init__(self):
        pass

    async def setup(self):
        if ImageHandler.pipe is None:
            print("Initializing ImageHandler...")
            ImageHandler.pipe = DiffusionPipeline.from_pretrained(
                os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir="/home/.cache/huggingface/hub",
                device_map="balanced"
            )
            ImageHandler.pipe.safety_checker = None
            print(f"ImageHandler initialized")

    async def generate_image(self, prompt: str) -> bytes:
        await self.setup()
        try:
            image = ImageHandler.pipe(prompt=prompt).images[0]
            return self._image_to_bytes(image)
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def _image_to_bytes(self, image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array.read()