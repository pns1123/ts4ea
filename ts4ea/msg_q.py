import asyncio
import numpy as np
import redis
import redis.asyncio as aredis

# --------------------------


class RedisConnection:
    """
    async context manager to hold connection to redis
    """

    def __init__(self, host="redis", port=6379):
        #prod
        #self.connection = redis.Redis(host=host, port=port)
        #dev
        self.connection = redis.Redis()

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(exc_type)
            print(exc_val)
            print(exc_tb)
        self.connection.close()


class AsyncRedisConnection:
    """
    async context manager to hold connection to redis
    """

    def __init__(self, host="redis", port=6379):
        #prod
        #self.connection = aredis.Redis(host=host, port=port)
        #dev
        self.connection = aredis.Redis()

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(exc_type)
            print(exc_val)
            print(exc_tb)
        await self.connection.close()


class AsyncRedisBroker:
    def __init__(self, redis_connection):
        self.started_tasks = []
        self.redis_connection = redis_connection

    async def read_new_items(self, stream_keys: dict):
        """
        asynchrously retrieve new entries from stream
        """
        print(f"listening for new messages in streams: {stream_keys}")
        try:
            return await self.redis_connection.xread(stream_keys, count=None, block=0)
        except ConnectionError as e:
            print("ERROR REDIS CONNECTION: {}".format(e))

    async def start_listening(self, stream_keys: dict, stream_processors: dict):
        """
        start a non-blocking loop to listen for and process new
        elements in the streams
        """

        while True:
            resp = await self.read_new_items(stream_keys)
            self.started_tasks += [
                asyncio.create_task(
                    stream_processors[stream_key](stamped_data)
                )
                for [stream_key, stamped_data] in resp
            ]
        asyncio.wait(self.started_tasks)

    async def send_data_to_stream(self, stream_name, data):
        try:
            resp = await self.redis_connection.xadd(stream_name, data)

        except ConnectionError as e:
            print("ERROR REDIS CONNECTION: {}".format(e))
