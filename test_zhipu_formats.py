"""Test Zhipu API with different JWT formats"""
import time
import jwt
import requests

key = '47487f4f03384d9bb3acb6edd234ff4f.JcyPPJwGY13TbEad'
api_key, secret = key.split('.')

ts = int(time.time() * 1000)

# Format 1: Standard
payload1 = {'api_key': api_key, 'exp': ts + 3600000, 'timestamp': ts}
token1 = jwt.encode(payload1, secret, algorithm='HS256')

# Format 2: No exp
payload2 = {'api_key': api_key, 'timestamp': ts}
token2 = jwt.encode(payload2, secret, algorithm='HS256')

# Format 3: Custom header
token3 = jwt.encode(payload1, secret, algorithm='HS256', headers={'alg': 'HS256', 'typ': 'JWT'})

url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
data = {'model': 'glm-5', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}

for i, tok in enumerate([token1, token2, token3], 1):
    try:
        r = requests.post(url, json=data, headers={'Authorization': f'Bearer {tok}'}, timeout=10)
        print(f'Format {i}: Status {r.status_code}')
        if r.status_code == 200:
            print(f'SUCCESS: {r.json()}')
            break
        else:
            print(f'  Response: {r.content}')
    except Exception as e:
        print(f'Format {i}: Error {e}')
