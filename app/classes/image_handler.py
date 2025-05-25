import os, torch, io, discord
from diffusers import DiffusionPipeline

class ImageHandler:
    pipe: DiffusionPipeline = None

    def __init__(self):
        pass

    async def setup(self):
        if ImageHandler.pipe is None:
            print("Initializing ImageHandler...")
            model = os.environ.get("IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            ImageHandler.pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, cache_dir="/home/.cache/huggingface/hub", safety_checker=None)
            ImageHandler.pipe.to("cuda")
            print(f"ImageHandler initialized with model: {model}")

    async def generate_image(self, prompt):
        await self.setup()
        try:
            image = ImageHandler.pipe(prompt).images[0]
            return self.return_discord_file(image)
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def return_discord_file(self, image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)  # Reset the stream position to the beginning             
        file = discord.File(byte_array, filename="generated_image.png")
        return file