from datetime import datetime

import json
import redis


def read_all_data(r: redis.Redis, cursor=0, data={}):
    cursor, key_list = r.scan(cursor)
    data = {
        **{
            key: r.xread({key: b"0-0"}) if r.type(key) == b"stream" else r.get(key)
            for key in key_list
        },
        **data,
    }
    if cursor == 0:
        return data
    else:
        return read_all_data(r, cursor, data)

def decode_data(data):
    try:
        match data:
            case list(data): return [decode_data(x) for x in data]
            case dict(data): return {decode_data(key): decode_data(val) for key, val in data.items()}
            case (a,b): return (decode_data(a), decode_data(b))
            case _: return data.decode()
    except:
        return ""

if __name__=="__main__":
    r = redis.Redis(host="redis", port=6379)
    all_data = read_all_data(r)
    no_bytes = {
        key: val
        for key, val in all_data.items()
    }
    
    decoded_data = decode_data(no_bytes)
    
    cur_time = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    with open(f"study_data_{cur_time}.json", 'w') as f:
        json.dump(decoded_data, f, sort_keys=True, indent=4)
