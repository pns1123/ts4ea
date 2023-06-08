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

