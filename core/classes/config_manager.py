import redis, os
class configManager:

    def __init__(self):
        self.redis = redis.Redis(host=os.environ['REDIS_HOST'], port=6379, db=0, charset="utf-8", decode_responses=True)
        self.namespace = "dcb"

    def get_setting(self, setting, guild_id):
        return self.redis.get(f"{self.namespace}:{guild_id}:{setting}") or False
    
    def update_setting(self, setting, value, guild_id):
        self.redis.set(f"{self.namespace}:{guild_id}:{setting}", value)
