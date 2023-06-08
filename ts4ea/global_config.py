import json

dev = False

if dev:
    REDIS_HOST = "0.0.0.0"
    SHARE = False
else:
    REDIS_HOST = "redis"
    SHARE = True

ATTENTION_CHECK_FILENAME = "BERLIN_ATTENTIONCHECK_77_berlin-ortsschild.png"

N_ROUNDS = 13

with open("streetview/filename2pred.json", "r") as f:
    FILENAME2PRED = json.load(f)
UNORDERED_FILENAMES = [
    f"{fn[:-4]}.png" for fn in FILENAME2PRED if fn != ATTENTION_CHECK_FILENAME
]
