import os, torch, io,gc
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class ImageToImageHandler:

    def __init__(self):
        pass

    async def setup(self):
        print("Initializing Img2Img pipeline...")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            pretrained_model_or_path=os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-refiner-1.0"),
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/home/.cache/huggingface/hub",
        )
        
        self.pipe.enable_model_cpu_offload()
        print("Img2Img pipeline ready.")

    async def image_to_image(self, prompt: str, init_image: Image.Image):
        await self.setup()
        try:
            output = self.pipe(prompt=prompt, image=init_image)
            return self.return_image_bytes(output.images[0])
        except Exception as e:
            print(f"Error during image-to-image generation: {e}")
            return None

    def release_resources(self):
        print("Releasing resources...")
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()


    def return_image_bytes(self, image: Image.Image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array