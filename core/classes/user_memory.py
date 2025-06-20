import redis
import json
import os
class UserMemory:
    def __init__(self, user_id, guild_id):
        self.user_id = user_id
        self.guild_id = guild_id
        self.redis = redis.Redis(host=os.environ['REDIS_HOST'], port=6379, db=0, charset="utf-8", decode_responses=True)
        self.key = f"guild:{self.guild_id}:user:{self.user_id}"

    def append(self, new_data):
        existing_data = self.get() or []
        existing_data.append(new_data)
        serialized = json.dumps(existing_data)
        self.redis.set(self.key, serialized)

    def get(self):
        value = self.redis.get(self.key)
        if value is not None:
            return json.loads(value)
        return None

    def remove(self, data):
        memories = self.get() or []
        if data in memories:
            memories.remove(data)
            serialized = json.dumps(memories)
            self.redis.set(self.key, serialized)
            return True
        return False

    def clear(self):
        self.redis.delete(self.key)