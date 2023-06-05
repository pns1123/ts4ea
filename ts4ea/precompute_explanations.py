import asyncio
import io
import json
import numpy as np
import os

from explainer import compute_render_explanation, predict, LIMEConfig, RenderConfig
from msg_q import AsyncRedisBroker, AsyncRedisConnection, RedisConnection
from PIL import Image


if __name__ == "__main__":
    filename2pred = {}
    for filename in os.listdir("streetview/raw/"):

        try:
            img = Image.open(f"streetview/raw/{filename}")
            pred = predict(img)
            exp_fixed = compute_render_explanation(img)
            filename2pred[filename] = {"pred": pred}
            img.save(f"streetview/original/{filename[:-4]}.png")
            exp_fixed.save(f"streetview/reference_explanations/{filename[:-4]}.png")
        except:
            continue

    with open("streetview/filename2pred.json", 'w') as f:
        json.dump(filename2pred, f, indent=4, sort_keys=True)
