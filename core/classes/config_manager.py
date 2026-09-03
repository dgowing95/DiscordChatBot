from classes.redis_client import text_client


class configManager:
    """Per-guild settings in Redis under the `dcb` namespace."""

    def __init__(self):
        self.namespace = "dcb"

    @property
    def redis(self):
        # Shared, lazily-built client (classes/redis_client.py). configManager
        # is constructed per message and per tool call, so building a pool per
        # instance opened a new connection each time and never closed it.
        return text_client()

    async def get_setting(self, setting, guild_id):
        return await self.redis.get(f"{self.namespace}:{guild_id}:{setting}") or False

    async def update_setting(self, setting, value, guild_id):
        await self.redis.set(f"{self.namespace}:{guild_id}:{setting}", value)
