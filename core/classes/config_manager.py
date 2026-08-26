import os
import redis.asyncio as redis
class configManager:

    def __init__(self):
        self.redis = redis.Redis(host=os.environ['REDIS_HOST'], port=6379, db=0, encoding="utf-8", decode_responses=True)
        self.namespace = "dcb"

    async def get_setting(self, setting, guild_id):
        return await self.redis.get(f"{self.namespace}:{guild_id}:{setting}") or False

    async def update_setting(self, setting, value, guild_id):
        await self.redis.set(f"{self.namespace}:{guild_id}:{setting}", value)
