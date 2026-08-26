import json
import os
import redis.asyncio as redis
class UserMemory:
    def __init__(self, user_id, guild_id):
        self.user_id = user_id
        self.guild_id = guild_id
        self.redis = redis.Redis(host=os.environ['REDIS_HOST'], port=6379, db=0, encoding="utf-8", decode_responses=True)
        self.key = f"guild:{self.guild_id}:user:{self.user_id}"

    async def append(self, new_data):
        existing_data = await self.get() or []
        existing_data.append(new_data)
        serialized = json.dumps(existing_data)
        await self.redis.set(self.key, serialized)

    async def get(self):
        value = await self.redis.get(self.key)
        if value is not None:
            json_data = json.loads(value)
            deduped = list(set(json_data))
            return deduped
        return None

    async def remove(self, data):
        memories = await self.get() or []
        if data in memories:
            memories.remove(data)
            serialized = json.dumps(memories)
            await self.redis.set(self.key, serialized)
            return True
        return False

    async def clear(self):
        await self.redis.delete(self.key)