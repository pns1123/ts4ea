import asyncio
import io
import json
import numpy as np
import os
import uuid

from configuration import ConfigurationEncoder
from dataclasses import asdict
from explainer import compute_render_explanation, predict, LIMEConfig, RenderConfig
from global_config import (
    ATTENTION_CHECK_FILENAME,
    CONFIG2ID,
    CONFIG_ENCODER,
    FILENAME2PRED,
    LEARNING_ROUNDS,
    MU_INITIAL,
    THOMPSON_SAMPLER,
    SIGMA2_INITIAL,
    UNORDERED_FILENAMES,
)
from itertools import cycle, islice
from msg_q import AsyncRedisBroker, AsyncRedisConnection, RedisConnection
from PIL import Image
from thompson_sampling import ThompsonSampler


def sample_config(thompson_sampler, mu, sigma2):
    theta = thompson_sampler.sample_model(mu, sigma2)
    params = thompson_sampler.select_arm(theta)

    lime_config = LIMEConfig(segmentation_method=params["segmentation_method"])
    render_config = RenderConfig(
        coverage=params["coverage"],
        opacity=params["opacity"],
    )
    return lime_config, render_config


def learning_setup(thompson_sampler, filename_iter, mu, sigma2):
    filename = next(filename_iter)

    img = Image.open(f"streetview/raw/{filename}")

    reference_params = thompson_sampler.select_arm(mu)
    reference_lime_config = LIMEConfig(
        segmentation_method=reference_params["segmentation_method"]
    )
    reference_render_config = RenderConfig(
        coverage=reference_params["coverage"],
        opacity=reference_params["opacity"],
    )

    candidate_lime_config, candidate_render_config = sample_config(
        thompson_sampler, mu, sigma2
    )
    return (
        filename,
        reference_lime_config,
        reference_render_config,
        candidate_lime_config,
        candidate_render_config,
    )


def evaluation_setup(thompson_sampler, filename_iter, mu, sigma2):
    filename = next(filename_iter)
    reference_lime_config, reference_render_config = LIMEConfig(), RenderConfig()
    candidate_params = thompson_sampler.select_arm(mu)
    candidate_lime_config = LIMEConfig(
        segmentation_method=candidate_params["segmentation_method"]
    )
    candidate_render_config = RenderConfig(
        coverage=candidate_params["coverage"],
        opacity=candidate_params["opacity"],
    )
    return (
        filename,
        reference_lime_config,
        reference_render_config,
        candidate_lime_config,
        candidate_render_config,
    )


def pred_correct(filename_label, pred_label):
    match pred_label:
        case "Berlin":
            return filename_label == "BERLIN"
        case "Hamburg":
            return filename_label == "HAMBURG"
        case "Tel Aviv":
            return filename_label == "TELAVIV"
        case "Jerusalem":
            return filename_label == "WESTJERUSALEM"


def sample_filename_iter():
    filename2pred_correct = {
        key: val
        for key, val in FILENAME2PRED.items()
        if pred_correct(key.split("_")[0], val)
    }
    filename2pred_incorrect = {
        key: val
        for key, val in FILENAME2PRED.items()
        if not pred_correct(key.split("_")[0], val)
    }
    # 0.75 correct and 0.25 incorrect prediction stratified sampling
    # 16 rounds -> 12 correct, 4 incorrect
    filenames_correct = np.random.choice(
        list(filename2pred_correct.keys()), size=12, replace=False
    )
    filenames_incorrect = np.random.choice(
        list(filename2pred_incorrect.keys()), size=4, replace=False
    )
    filenames_total = np.concatenate([filenames_correct, filenames_incorrect])
    np.random.shuffle(filenames_total)
    # attention check during evaluation: show same image/explanation twice
    filenames_total[15] = filenames_total[11]

    return list(filenames_total)


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
        filename_iter = islice(cycle(filename_permutation), current_round - 1, None)


        match current_round:
            # start evaluation after LEARNING_ROUNDS
            case current_round if current_round >= LEARNING_ROUNDS:
                mu_posterior = np.array(param_dict.get("mu"))
                sigma2_posterior = np.array(param_dict.get("sigma2"))
                (
                    filename,
                    reference_lime_config,
                    reference_render_config,
                    candidate_lime_config,
                    candidate_render_config,
                ) = evaluation_setup(
                    THOMPSON_SAMPLER, filename_iter, mu_posterior, sigma2_posterior
                )
            # learning setup during rounds 0..(LEARNING_ROUNDS-1)
            case _:
                mu_posterior, sigma2_posterior = THOMPSON_SAMPLER.amp_update(
                    np.array(param_dict.get("mu")),
                    np.array(param_dict.get("sigma2")),
                    np.array(
                        json.loads(interaction_data[b"candidate_feature_vec"])["coef"]
                    ),
                    float(interaction_data.get(b"reward").decode()),
                )

                (
                    filename,
                    reference_lime_config,
                    reference_render_config,
                    candidate_lime_config,
                    candidate_render_config,
                ) = learning_setup(
                    THOMPSON_SAMPLER, filename_iter, mu_posterior, sigma2_posterior
                )

        async with AsyncRedisConnection() as conn:

            await conn.set(
                parameter_key,
                json.dumps(
                    {"mu": list(mu_posterior), "sigma2": list(sigma2_posterior)}
                ),
            )

            await conn.xadd(
                str.encode(f"{self.user_id.decode()}_explanations"),
                {
                    "filename": filename,
                    "pred": FILENAME2PRED[filename],
                    "round": current_round,
                    "explanation_id": str(uuid.uuid4()),
                    "reference_feature_vec": json.dumps(
                        {
                            "coef": list(
                                CONFIG_ENCODER.encode(
                                    {
                                        **asdict(reference_lime_config),
                                        **asdict(reference_render_config),
                                    }
                                )
                            )
                        }
                    ),
                    "candidate_feature_vec": json.dumps(
                        {
                            "coef": list(
                                CONFIG_ENCODER.encode(
                                    {
                                        **asdict(candidate_lime_config),
                                        **asdict(candidate_render_config),
                                    }
                                )
                            )
                        }
                    ),
                    "model_params": json.dumps(
                        {"mu": list(mu_posterior), "sigma2": list(sigma2_posterior)}
                    ),
                },
            )


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
    user_filename_seq = sample_filename_iter()
    async with AsyncRedisConnection() as conn:
        permutation_key = str.encode(f"{user_id.decode()}_filename_permutation")
        await conn.set(
            permutation_key,
            json.dumps({"filename_permutation": user_filename_seq}),
        )
        parameter_key = str.encode(f"{user_id.decode()}_parameter")
        await conn.set(
            parameter_key,
            json.dumps({"mu": list(MU_INITIAL), "sigma2": list(SIGMA2_INITIAL)}),
        )

    # keep track of displayed images in redis
    filename_iter = cycle(user_filename_seq)

    # load static images
    (
        filename,
        reference_lime_config,
        reference_render_config,
        candidate_lime_config,
        candidate_render_config,
    ) = learning_setup(THOMPSON_SAMPLER, filename_iter, MU_INITIAL, SIGMA2_INITIAL)

    # stream_name = b"test_stream"
    async with AsyncRedisConnection() as conn:
        await conn.xadd(
            str.encode(f"{user_id.decode()}_explanations"),
            {
                "filename": filename,
                "pred": FILENAME2PRED[filename],
                "round": 1,
                "explanation_id": str(uuid.uuid4()),
                "reference_feature_vec": json.dumps(
                    {
                        "coef": list(
                            CONFIG_ENCODER.encode(
                                {
                                    **asdict(reference_lime_config),
                                    **asdict(reference_render_config),
                                }
                            )
                        )
                    }
                ),
                "candidate_feature_vec": json.dumps(
                    {
                        "coef": list(
                            CONFIG_ENCODER.encode(
                                {
                                    **asdict(candidate_lime_config),
                                    **asdict(candidate_render_config),
                                }
                            )
                        )
                    }
                ),
                "model_params": json.dumps(
                    {"mu": list(MU_INITIAL), "sigma2": list(SIGMA2_INITIAL)}
                ),
            },
        )


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
