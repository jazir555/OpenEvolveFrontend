"""Test Zhipu API - try different endpoints and models"""
import time
import jwt
import requests

key = '9266cf5bd6d845f4b0e515ee3b5698a0.IouaWjdtuPNGEYyW'
api_key, secret = key.split('.')

ts = int(time.time() * 1000)
payload = {'api_key': api_key, 'exp': ts + 3600000, 'timestamp': ts}
token = jwt.encode(payload, secret, algorithm='HS256')

headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

# Try different endpoints and models
tests = [
    ('v4 + glm-4', 'https://open.bigmodel.cn/api/paas/v4/chat/completions', {'model': 'glm-4', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}),
    ('v4 + glm-3-turbo', 'https://open.bigmodel.cn/api/paas/v4/chat/completions', {'model': 'glm-3-turbo', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}),
    ('v4 + chatglm_turbo', 'https://open.bigmodel.cn/api/paas/v4/chat/completions', {'model': 'chatglm_turbo', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}),
]

print("Testing Zhipu API endpoints...\n")

for name, url, data in tests:
    try:
        r = requests.post(url, json=data, headers=headers, timeout=15)
        print(f'{name}: {r.status_code}')
        if r.status_code == 200:
            print(f'  SUCCESS: {r.json()}')
        else:
            try:
                err = r.json()
                msg = err.get('error', {}).get('message', '')
                # Try to show error
                if msg:
                    print(f'  Error: {msg[:100]}')
            except:
                pass
    except Exception as e:
        print(f'{name}: Exception - {e}')
    print()
