import os, torch, io, gc
from diffusers import DiffusionPipeline

class TextToImageHandler:
    pipe: DiffusionPipeline = None

    def __init__(self):
        pass

    async def setup(self):
        if TextToImageHandler.pipe is None:
            print("Initializing ImageHandler...")
            TextToImageHandler.pipe = DiffusionPipeline.from_pretrained(
                os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir="/home/.cache/huggingface/hub",
                device_map="balanced"
            )
            TextToImageHandler.pipe.safety_checker = None
            print(f"ImageHandler initialized")

    async def generate_image(self, prompt: str) -> bytes:
        await self.setup()
        try:
            image = TextToImageHandler.pipe(prompt=prompt).images[0]
            self.release_resources()
            return self._image_to_bytes(image)
        except Exception as e:
            print(f"Error generating image: {e}")
            self.release_resources()
            return None

    def release_resources(self):
        if TextToImageHandler.pipe is not None:
            print("Releasing resources...")
            del TextToImageHandler.pipe
            TextToImageHandler.pipe = None
            torch.cuda.empty_cache()
            gc.collect()

    def _image_to_bytes(self, image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array.read()