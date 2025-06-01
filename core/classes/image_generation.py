import aiohttp, os, discord
async def generate_image_from_api(prompt: str) -> bytes:
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        async with session.post(
            f"http://{os.environ.get("DIFFUSION_URL", 5)}:8000/text-image",
            json={"prompt": prompt}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"API failed with status {resp.status}")
            return await resp.read()
    

async def modify_image_from_api(prompt: str, image: discord.Attachment) -> bytes:
    async with aiohttp.ClientSession(auto_decompress=False) as session:
        image_bytes = await image.read()
        data = aiohttp.FormData()
        data.add_field('prompt', prompt)
        data.add_field('file', image_bytes)

        async with session.post(
            f"http://{os.environ.get("DIFFUSION_URL", 5)}:8000/image-image",
            data=data
        ) as resp:
            if resp.status != 200:
                raise Exception(f"API failed with status {resp.status}")
            return await resp.read()