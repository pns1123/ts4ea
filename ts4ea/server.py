import asyncio
import io
import json
import numpy as np
import os

from configuration import ConfigurationEncoder
from dataclasses import asdict
from explainer import compute_render_explanation, predict, LIMEConfig, RenderConfig
from global_config import ATTENTION_CHECK_FILENAME
from itertools import cycle, islice
from msg_q import AsyncRedisBroker, AsyncRedisConnection, RedisConnection
from PIL import Image
from thompson_sampling import ThompsonSampler

# ------------------------------------------------------------
with open("streetview/filename2pred.json", "r") as f:
    FILENAME2PRED = json.load(f)
UNORDERED_FILENAMES = [
    f"{fn[:-4]}.png" for fn in FILENAME2PRED if fn != ATTENTION_CHECK_FILENAME
]

categorical_variables = {
    "segmentation_method": ["felzenszwalb", "slic"],  # , "quickshift", "watershed"],
    "coverage": [0.15, 0.5, 0.85],
    "opacity": [0.15, 0.5, 0.85],
}

config_encoder = ConfigurationEncoder(categorical_variables, {})
n_var = config_encoder.categorical_offset[-1] + len(config_encoder.numerical_variables)

thompson_sampler = ThompsonSampler(config_encoder=config_encoder)
# ------------------------------------------------------------


class ExplanationDistributor:
    def __init__(self, user_id):
        self.user_id = user_id

    async def send_explanations(self, stream_res):
        timestamp, interaction_data = stream_res[0]

        # fix a permutation of filenames for user and store list in redis
        async with AsyncRedisConnection() as conn:
            current_round = (
                await conn.xlen(str.encode(f"{self.user_id.decode()}_reward_history"))
                + 1
            )
            permutation_key = str.encode(
                f"{self.user_id.decode()}_filename_permutation"
            )
            filename_permutation = json.loads(
                ((await conn.get(permutation_key)).decode())
            ).get("filename_permutation")

            parameter_key = str.encode(f"{self.user_id.decode()}_parameter")
            param_dict = json.loads((await conn.get(parameter_key)).decode())

        # keep track of displayed images in redis
        filename_iter = islice(cycle(filename_permutation), current_round, None)

        # open buffers
        img_buffer = io.BytesIO()
        ref_exp_buffer = io.BytesIO()
        exp_adjusted_buffer = io.BytesIO()

        mu_posterior, sigma2_posterior = thompson_sampler.amp_update(
            np.array(param_dict.get("mu")),
            np.array(param_dict.get("sigma2")),
            np.array(json.loads(interaction_data.get(b"feature_vec").decode())["coef"]),
            float(interaction_data.get(b"reward").decode()),
        )

        async with AsyncRedisConnection() as conn:
            await conn.set(
                parameter_key,
                json.dumps(
                    {"mu": list(mu_posterior), "sigma2": list(sigma2_posterior)}
                ),
            )

        theta = thompson_sampler.sample_model(mu_posterior, sigma2_posterior)
        params = thompson_sampler.select_arm(theta)

        lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])
        render_config = RenderConfig(
            coverage=params["coverage"],
            opacity=params["opacity"],
        )

        if current_round == 6:
            filename = ATTENTION_CHECK_FILENAME
            img = Image.open(f"streetview/raw/{filename}")
            exp_adjusted = compute_render_explanation(
                img, lime_config=lime_config, render_config=render_config
            )

        else:
            filename = next(filename_iter)
            img = Image.open(f"streetview/raw/{filename}")

            try:
                exp_adjusted = compute_render_explanation(
                    img, lime_config=lime_config, render_config=render_config
                )
            except:
                print("exception in explanation computation")
                # load static images
                exp_adjusted = img

        img.save(img_buffer, format="PNG")
        ref_exp = Image.open(f"streetview/reference_explanations/{filename}")
        ref_exp.save(ref_exp_buffer, format="PNG")
        exp_adjusted.save(exp_adjusted_buffer, format="PNG")

        async with AsyncRedisConnection() as conn:
            await conn.xadd(
                str.encode(f"{self.user_id.decode()}_explanations"),
                {
                    "img_bytes": img_buffer.getvalue(),
                    "ref_exp_bytes": ref_exp_buffer.getvalue(),
                    "exp_adjusted_bytes": exp_adjusted_buffer.getvalue(),
                    "pred": FILENAME2PRED[filename]["pred"],
                    "round": current_round,
                    "feature_vec": json.dumps(
                        {"coef": list(config_encoder.encode(params))}
                    ),
                },
            )
            await conn.xadd(
                str.encode(f"{self.user_id.decode()}_explanation_infos"),
                {
                    "filename": filename,
                    "pred": FILENAME2PRED[filename]["pred"],
                    "round": current_round,
                    "feature_vec": json.dumps(
                        {"coef": list(config_encoder.encode(params))}
                    ),
                    "model_params": json.dumps(
                        {"mu": list(mu_posterior), "sigma2": list(sigma2_posterior)}
                    ),
                },
            )

        img_buffer.close()
        ref_exp_buffer.close()
        exp_adjusted_buffer.close()


async def register_user(stream_res):
    timestamp, msg = stream_res[0]
    user_id = msg.get(b"user_id")

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

    # fix a permutation of filenames for user and store list in redis
    user_filename_permutation = UNORDERED_FILENAMES.copy()
    np.random.shuffle(user_filename_permutation)
    filename_iter = cycle(user_filename_permutation)
    default_params = {**asdict(LIMEConfig()), **asdict(RenderConfig())}
    async with AsyncRedisConnection() as conn:
        permutation_key = str.encode(f"{user_id.decode()}_filename_permutation")
        await conn.set(
            permutation_key,
            json.dumps({"filename_permutation": user_filename_permutation}),
        )
        parameter_key = str.encode(f"{user_id.decode()}_parameter")
        mu = 0.5 * config_encoder.encode(
            {
                key: default_params[key]
                for key in config_encoder.decode(config_encoder.sample_feature())
            }
        )
        sigma2 = 0.5 * np.ones(len(mu))
        await conn.set(
            parameter_key,
            json.dumps({"mu": list(mu), "sigma2": list(sigma2)}),
        )

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

    params = thompson_sampler.sample_model(mu, sigma2)
    default_params = {**asdict(LIMEConfig()), **asdict(RenderConfig())}

    theta = thompson_sampler.sample_model(mu, sigma2)
    params = thompson_sampler.select_arm(theta)

    lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])
    render_config = RenderConfig(
        coverage=params["coverage"],
        opacity=params["opacity"],
    )

    while all([params[key] == default_params[key] for key in params]):
        params = config_encoder.decode(config_encoder.sample_feature())

    lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])

    render_config = RenderConfig(
        coverage=params["coverage"],
        opacity=params["opacity"],
    )
    exp_adjusted = compute_render_explanation(
        img, lime_config=lime_config, render_config=render_config
    )
    exp_adjusted.save(exp_adjusted_buffer, format="PNG")

    # stream_name = b"test_stream"
    async with AsyncRedisConnection() as conn:
        await conn.xadd(
            str.encode(f"{user_id.decode()}_explanations"),
            {
                "img_bytes": img_buffer.getvalue(),
                "ref_exp_bytes": ref_exp_buffer.getvalue(),
                "exp_adjusted_bytes": exp_adjusted_buffer.getvalue(),
                "pred": FILENAME2PRED[filename]["pred"],
                "round": 1,
                "feature_vec": json.dumps(
                    {"coef": list(config_encoder.encode(params))}
                ),
                "model_params": json.dumps({"mu": list(mu), "sigma2": list(sigma2)}),
            },
        )
        await conn.xadd(
            str.encode(f"{user_id.decode()}_explanation_infos"),
            {
                "filename": filename,
                "pred": FILENAME2PRED[filename]["pred"],
                "round": 0,
                "feature_vec": json.dumps(
                    {"coef": list(config_encoder.encode(params))}
                ),
            },
        )

    img_buffer.close()
    ref_exp_buffer.close()
    exp_adjusted_buffer.close()


async def ping(msg):
    return


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
