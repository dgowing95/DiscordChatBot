import os, torch, io, discord
from diffusers import DiffusionPipeline

class ImageHandler:
    pipe: DiffusionPipeline = None
    refiner : DiffusionPipeline = None
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps: int = 40
    high_noise_frac: float = 0.8

    def __init__(self):
        pass

    async def setup(self):
        if ImageHandler.pipe is None:
            print("Initializing ImageHandler...")

            # load both base & refiner
            ImageHandler.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir="/home/.cache/huggingface/hub", safety_checker=None
            )
            ImageHandler.pipe.enable_model_cpu_offload()
            ImageHandler.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=ImageHandler.pipe.text_encoder_2,
                cache_dir="/home/.cache/huggingface/hub",
                vae=ImageHandler.pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            ImageHandler.refiner.enable_model_cpu_offload()


            print(f"ImageHandler initialized")

    async def generate_image(self, prompt):
        await self.setup()
        try:
            image = ImageHandler.pipe(
                prompt=prompt,
                num_inference_steps=ImageHandler.n_steps,
                denoising_end=ImageHandler.high_noise_frac,
                output_type="latent",
            ).images[0]
            image = ImageHandler.refiner(
                prompt=prompt,
                num_inference_steps=ImageHandler.n_steps,
                denoising_start=ImageHandler.high_noise_frac,
                image=image,
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