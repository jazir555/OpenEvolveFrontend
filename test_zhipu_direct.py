"""Test Zhipu AI (Z.ai) API with correct auth"""
import time
import jwt
import requests
import hashlib

key = '9555ab1f79be4db498d0cf19c1277af0.8lf475NPrUgEWaF0'
api_key, secret = key.split('.')

print(f'API Key: {api_key}')
print(f'Secret: {secret}')

# Method 1: Standard JWT
print('\n=== Method 1: Standard JWT ===')
timestamp = int(time.time() * 1000)
payload = {
    'api_key': api_key,
    'exp': timestamp + 3600000,
    'timestamp': timestamp
}
token = jwt.encode(payload, secret, algorithm='HS256')

url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
data = {'model': 'glm-5', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 20}

response = requests.post(url, json=data, headers=headers, timeout=30)
print(f'Status: {response.status_code}')
print(f'Response: {response.content.decode("utf-8", errors="replace")}')

# Method 2: API Key + Secret with signature
print('\n=== Method 2: Signature Auth ===')
timestamp = int(time.time() * 1000)
sign_str = f'{api_key}{timestamp}'
signature = hashlib.sha256(f'{sign_str}{secret}'.encode()).hexdigest()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
data = {
    'model': 'glm-5',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 20,
    'timestamp': timestamp,
    'signature': signature
}

response = requests.post(url, json=data, headers=headers, timeout=30)
print(f'Status: {response.status_code}')
print(f'Response: {response.content.decode("utf-8", errors="replace")}')
