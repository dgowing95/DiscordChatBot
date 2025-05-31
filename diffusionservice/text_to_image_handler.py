import os, torch, io, gc
from diffusers import DiffusionPipeline

class TextToImageHandler:
    def _init__(self):
        pass

    async def setup(self):
        print("Initializing ImageHandler...")
        self.pipe = DiffusionPipeline.from_pretrained(
            os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="/home/.cache/huggingface/hub",
            device_map="balanced"
        )
        self.pipe.safety_checker = None
        print(f"ImageHandler initialized")

    async def generate_image(self, prompt: str) -> bytes:
        await self.setup()
        try:
            image = self.pipe(prompt=prompt).images[0]
            return self._image_to_bytes(image)
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def release_resources(self):
        print("Releasing resources...")
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()

    def _image_to_bytes(self, image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array.read()