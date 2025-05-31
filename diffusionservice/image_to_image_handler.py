import os, torch, io,gc
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class ImageToImageHandler:
    pipe: AutoPipelineForImage2Image = None

    def __init__(self):
        pass

    async def setup(self):
        if ImageToImageHandler.pipe is None:
            print("Initializing Img2Img pipeline...")
            ImageToImageHandler.pipe = AutoPipelineForImage2Image.from_pretrained(
                pretrained_model_or_path=os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-refiner-1.0"),
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir="/home/.cache/huggingface/hub",
            )
            ImageToImageHandler.pipe.enable_model_cpu_offload()
            print("Img2Img pipeline ready.")

    async def image_to_image(self, prompt: str, init_image: Image.Image):
        await self.setup()
        try:
            output = ImageToImageHandler.pipe(prompt=prompt, image=init_image)
            #self.release_resources()
            return self.return_image_bytes(output.images[0])
        except Exception as e:
            print(f"Error during image-to-image generation: {e}")
            #self.release_resources()
            return None

    def release_resources(self):
        if ImageToImageHandler.pipe is not None:
            print("Releasing resources...")
            del ImageToImageHandler.pipe
            ImageToImageHandler.pipe = None
            torch.cuda.empty_cache()
            gc.collect()


    def return_image_bytes(self, image: Image.Image):
        byte_array = io.BytesIO()
        image.save(byte_array, format='PNG')
        byte_array.seek(0)
        return byte_array