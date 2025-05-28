import os, torch, io, discord
from diffusers import DiffusionPipeline

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
            print(ImageHandler.pipe.hf_device_map)

            ImageHandler.pipe.safety_checker = None


            print(f"ImageHandler initialized")

    async def generate_image(self, prompt):
        await self.setup()
        try:
            image = ImageHandler.pipe(
                prompt=prompt
            ).images[0]
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