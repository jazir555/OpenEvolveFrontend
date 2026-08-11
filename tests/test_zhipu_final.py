"""Test Zhipu AI with guaranteed working key"""
import time
import jwt
import requests
import sys

# New guaranteed working key
key = '9266cf5bd6d845f4b0e515ee3b5698a0.IouaWjdtuPNGEYyW'
api_key, secret = key.split('.')

print(f'Testing key: {api_key}', flush=True)
print(f'Secret: {secret}', flush=True)

# Zhipu uses timestamp in milliseconds
ts = int(time.time() * 1000)

# Payload per Zhipu docs
payload = {
    'api_key': api_key,
    'exp': ts + 3600000,  # 1 hour
    'timestamp': ts
}

# Create JWT
token = jwt.encode(payload, secret, algorithm='HS256')
print(f'\nJWT Token: {token[:80]}...', flush=True)

# Make request
url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}
data = {
    'model': 'glm-5',
    'messages': [{'role': 'user', 'content': 'Hello, respond in English'}],
    'max_tokens': 50,
    'stream': False
}

print(f'\nSending POST to {url}...', flush=True)

try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    
    print(f'\nResponse Status: {response.status_code}', flush=True)
    
    if response.status_code == 200:
        result = response.json()
        print(f'\n*** SUCCESS ***', flush=True)
        print(f'Content: {result["choices"][0]["message"]["content"]}', flush=True)
        print(f'Tokens: {result.get("usage", {})}', flush=True)
    else:
        # Get error message
        try:
            error_data = response.json()
            msg = error_data.get('error', {}).get('message', 'Unknown error')
            # Decode Chinese if present
            if isinstance(msg, str):
                print(f'Error: {msg}', flush=True)
        except Exception as e:
            print(f'Error parsing response: {e}', flush=True)
            print(f'Raw: {response.content}', flush=True)
            
except requests.exceptions.Timeout:
    print('ERROR: Request timed out', flush=True)
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}', flush=True)
