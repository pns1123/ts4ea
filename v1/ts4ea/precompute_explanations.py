import asyncio
import io
import json
import numpy as np
import os

from dataclasses import asdict
#from explainer import compute_render_explanation, predict, LIMEConfig, RenderConfig
from explainer import LIMEConfig, RenderConfig
from global_config import PARAMETER_LEVELS
from hashlib import sha256
from PIL import Image

configs = [
    {**asdict(LIMEConfig(**{"segmentation_method": sm})), 
     **asdict(RenderConfig(**{"coverage": cov, "opacity": op}))}
    for sm in PARAMETER_LEVELS["segmentation_method"]
    for cov in PARAMETER_LEVELS["coverage"]
    for op in PARAMETER_LEVELS["opacity"]
]

if __name__ == "__main__":
    config2id = {}
    filename2pred = {}
    for i, c in enumerate(configs):
        print(f"ID: {i}, c: {tuple(sorted(c.items()))}", 
                f"\nh: {sha256(str((tuple(sorted(c.items())))).encode()).hexdigest()}")
        config2id[sha256(str((tuple(sorted(c.items())))).encode()).hexdigest()] = i
        #lime_config = LIMEConfig()
        #render_config = RenderConfig()
        #lime_config = LIMEConfig(segmentation_method=c["segmentation_method"])
        #render_config = RenderConfig(
        #    coverage=c["coverage"],
        #    opacity=c["opacity"],
        #)

        #for filename in os.listdir("streetview/raw/"):
        #    try:
        #        img = Image.open(f"streetview/raw/{filename}")
        #        pred = predict(img)
        #        filename2pred[filename] = pred

        #        #exp_fixed = compute_render_explanation(img, lime_config, render_config)
        #        #exp_fixed.save(f"streetview/exp_conf_{i}/{filename[:-4]}.png")
        #    except ZeroDivisionError:
        #        print(f"FAILED: i={i}, fn={filename}")
        #        continue

    #with open("streetview/config2id.json", 'w') as f:
    #    json.dump(config2id, f, indent=4, sort_keys=True)


    #with open("streetview/filename2pred.json", 'w') as f:
    #    json.dump(filename2pred, f, indent=4, sort_keys=True)
