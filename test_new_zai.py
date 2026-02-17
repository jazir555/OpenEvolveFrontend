"""Test NEW Z.ai API Key"""
import time
import jwt
import requests

# New Z.ai key
key = '47487f4f03384d9bb3acb6edd234ff4f.JcyPPJwGY13TbEad'
api_key, secret = key.split('.')

print(f'Testing NEW Z.ai key: {api_key[:10]}...')

# JWT Auth
timestamp = int(time.time() * 1000)
payload = {
    'api_key': api_key,
    'exp': timestamp + 3600000,
    'timestamp': timestamp
}
token = jwt.encode(payload, secret, algorithm='HS256')

print(f'Token: {token[:50]}...')

url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
data = {'model': 'glm-5', 'messages': [{'role': 'user', 'content': 'Say hello in English'}], 'max_tokens': 20}

print(f'Sending request...')
try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f'Status: {response.status_code}')
    
    if response.status_code == 200:
        result = response.json()
        print('SUCCESS!')
        print(f'Response: {result["choices"][0]["message"]["content"]}')
    else:
        content = response.content.decode('utf-8', errors='replace')
        print(f'Error response: {content}')
except Exception as e:
    print(f'Exception: {e}')
