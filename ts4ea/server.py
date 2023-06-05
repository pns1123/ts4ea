import asyncio
import io
import json
import numpy as np
import os

from configuration import ConfigurationEncoder
from dataclasses import asdict
from explainer import compute_render_explanation, predict, LIMEConfig, RenderConfig
from itertools import cycle
from msg_q import AsyncRedisBroker, AsyncRedisConnection, RedisConnection
from PIL import Image
from thompson_sampling import ThompsonSampler


with open("streetview/filename2pred.json", "r") as f:
    filename2pred = json.load(f)

filenames = [f"{fn[:-4]}.png" for fn in filename2pred]
np.random.shuffle(filenames)
filename_iter = cycle(filenames)

categorical_variables = {
    "segmentation_method": ["felzenszwalb", "slic"],
    "negative": [None, "darkblue"],
    # "coverage": [0.15, 0.5, 0.85],
    # "opacity": [0.15, 0.5, 0.85],
}

# numerical_variables = {"coverage": (0, 1), "opacity": (0, 1)}
numerical_variables = {}

config_encoder = ConfigurationEncoder(categorical_variables, numerical_variables)
n_var = config_encoder.categorical_offset[-1] + len(config_encoder.numerical_variables)

thompson_sampler = ThompsonSampler(config_encoder=config_encoder)


class RoundCounter:
    def __init__(self):
        self.cur_round = 0

    def get(self):
        return self.cur_round

    def update(self):
        self.cur_round = (self.cur_round + 1) % 10


cur_round = RoundCounter()


class ExplanationDistributor:
    def __init__(self, user_id):
        self.user_id = user_id

    async def send_explanations(self, stream_res):
        print("send_explanation called")
        timestamp, msg = stream_res[0]

        # open buffers
        img_buffer = io.BytesIO()
        ref_exp_buffer = io.BytesIO()
        exp_adjusted_buffer = io.BytesIO()

        # load static images
        filename = next(filename_iter)
        img = Image.open(f"streetview/raw/{filename}")
        img.save(img_buffer, format="PNG")

        ref_exp = Image.open(f"streetview/reference_explanations/{filename}")
        ref_exp.save(ref_exp_buffer, format="PNG")

        # compute adjusted explanation
        stream_name = str.encode(f"{self.user_id.decode()}_reward_history")
        async with AsyncRedisConnection() as conn:
            [[_, reward_history]] = await conn.xread({stream_name: b"0-0"})

        data_dict_list = [data_dict for _, data_dict in reward_history]
        X = np.array(
            [
                json.loads(data_dict[b"feature_vec"].decode())["coef"]
                for data_dict in data_dict_list
            ]
        )
        y = np.array([float(data_dict[b"reward"].decode()) for data_dict in data_dict_list])
        theta = thompson_sampler.sample_model(X, y)
        params = thompson_sampler.select_arm(theta)

        lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])
        render_config = RenderConfig(
            #    coverage=params["coverage"],
            #    opacity=params["opacity"],
            negative=params["negative"],
        )
        exp_adjusted = compute_render_explanation(
            img, lime_config=lime_config, render_config=render_config
        )
        exp_adjusted.save(exp_adjusted_buffer, format="PNG")

        stream_name = str.encode(f"{self.user_id.decode()}_explanations")
        print(f"send_images: xadd to {stream_name}")
        async with AsyncRedisConnection() as conn:
            await conn.xadd(
                stream_name,
                {
                    "img_bytes": img_buffer.getvalue(),
                    "ref_exp_bytes": ref_exp_buffer.getvalue(),
                    "exp_adjusted_bytes": exp_adjusted_buffer.getvalue(),
                    "pred": filename2pred[filename]["pred"],
                    "round": cur_round.get() + 1,
                    "feature_vec": json.dumps(
                        {"coef": list(config_encoder.encode(params))}
                    ),
                },
            )

        img_buffer.close()
        ref_exp_buffer.close()
        exp_adjusted_buffer.close()
        cur_round.update()
        if cur_round.get() == 9:
            conn.xtrim("default_user", 0)


async def register_user(stream_res):
    timestamp, msg = stream_res[0]
    user_id = msg.get(b"user_id")
    print(f"register_user: user_id = {user_id}")
    if user_id is not None:
        stream_name = str.encode(f"{user_id.decode()}_reward_history")
        stream_keys[stream_name] = b"$"
        stream_keys[b"hello"] = b"$"
        stream_processors[stream_name] = ExplanationDistributor(
            user_id
        ).send_explanations

    async with AsyncRedisConnection() as conn:
        # ping the msg q for changes in stream key to take effect
        # o/w msg q could wait for new messages based on non-updated stream keys
        await conn.xadd(b"ping", {"msg": "pong"})
        print("xadd ping called")

    # open buffers
    img_buffer = io.BytesIO()
    ref_exp_buffer = io.BytesIO()
    exp_adjusted_buffer = io.BytesIO()

    # load static images
    filename = next(filename_iter)
    img = Image.open(f"streetview/raw/{filename}")
    img.save(img_buffer, format="PNG")

    ref_exp = Image.open(f"streetview/reference_explanations/{filename}")
    ref_exp.save(ref_exp_buffer, format="PNG")

    # 1st model is uniformly random drawn from configs
    params_feature = config_encoder.sample_feature()
    params = config_encoder.decode(params_feature)
    default_params = {**asdict(LIMEConfig()), **asdict(RenderConfig())}

    while all([params[key] == default_params[key] for key in params]):
        params = config_encoder.decode(config_encoder.sample_feature())

    lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])

    render_config = RenderConfig(
        #    coverage=params["coverage"],
        #    opacity=params["opacity"],
        negative=params["negative"],
    )
    exp_adjusted = compute_render_explanation(
        img, lime_config=lime_config, render_config=render_config
    )
    exp_adjusted.save(exp_adjusted_buffer, format="PNG")

    stream_name = str.encode(f"{user_id.decode()}_explanations")
    # stream_name = b"test_stream"
    async with AsyncRedisConnection() as conn:
        print(f"register_user: xadd to {stream_name}")
        await conn.xadd(
            stream_name,
            {
                "img_bytes": img_buffer.getvalue(),
                "ref_exp_bytes": ref_exp_buffer.getvalue(),
                "exp_adjusted_bytes": exp_adjusted_buffer.getvalue(),
                "pred": filename2pred[filename]["pred"],
                "round": cur_round.get() + 1,
                "feature_vec": json.dumps(
                    {"coef": list(config_encoder.encode(params))}
                ),
            },
        )

    img_buffer.close()
    ref_exp_buffer.close()
    exp_adjusted_buffer.close()
    cur_round.update()

async def ping(msg):
    print("pong")


stream_keys = {
    b"hello": b"$",
    b"ping": b"$",
}
stream_processors = {
    b"hello": register_user,
    b"ping": ping,
}


async def listen(stream_keys, stream_processors):
    async with AsyncRedisConnection() as conn:
        redis_broker = AsyncRedisBroker(conn)
        await redis_broker.start_listening(stream_keys, stream_processors)


if __name__ == "__main__":
    asyncio.run(listen(stream_keys, stream_processors))
