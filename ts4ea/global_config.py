import json
import numpy as np

from configuration import ConfigurationEncoder, LIMEConfig, RenderConfig
from dataclasses import asdict
from thompson_sampling import ThompsonSampler

# --------------------------------------------------------------
# REDIS PARAMS
local = False

if local:
    REDIS_HOST = "0.0.0.0"
else:
    REDIS_HOST = "redis"

SHARE = False

# --------------------------------------------------------------
# FILES
ATTENTION_CHECK_FILENAME = "BERLIN_ATTENTIONCHECK_77_berlin-ortsschild.png"

with open("streetview/filename2pred.json", "r") as f:
    FILENAME2PRED = json.load(f)
UNORDERED_FILENAMES = [
    f"{fn[:-4]}.png" for fn in FILENAME2PRED if fn != ATTENTION_CHECK_FILENAME
]

with open("streetview/config2id.json", "r") as f:
    CONFIG2ID = json.load(f)

# --------------------------------------------------------------
# STUDY PARAMETERS
LEARNING_ROUNDS = 11
EVAL_ROUNDS = 5
TOTAL_ROUNDS = LEARNING_ROUNDS + EVAL_ROUNDS

PARAMETER_LEVELS = {
    "segmentation_method": ["felzenszwalb", "slic"],  # , "quickshift", "watershed"],
    "coverage": [0.15, 0.5, 0.85],
    "opacity": [0.15, 0.5, 0.85],
}

# --------------------------------------------------------------
# THOMPSON SAMPLER
CONFIG_ENCODER = ConfigurationEncoder(PARAMETER_LEVELS, {})
THOMPSON_SAMPLER = ThompsonSampler(config_encoder=CONFIG_ENCODER)


# --------------------------------------------------------------
# INITIAL MODEL PARAMETERS
default_params = {**asdict(LIMEConfig()), **asdict(RenderConfig())}
MU_INITIAL = 0.5 * CONFIG_ENCODER.encode(
    {
        key: default_params[key]
        for key in CONFIG_ENCODER.decode(CONFIG_ENCODER.sample_feature())
    }
)

#MU_INITIAL = np.zeros(len(CONFIG_ENCODER.sample_feature()))
SIGMA2_INITIAL = 0.5*np.ones(len(MU_INITIAL))

