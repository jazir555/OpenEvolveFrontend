"""Test Zhipu AI with official SDK"""
from zhipuai import ZhipuAI

client = ZhipuAI(api_key='9266cf5bd6d845f4b0e515ee3b5698a0.IouaWjdtuPNGEYyW')

print('Testing with official ZhipuAI SDK...')
try:
    response = client.chat.completions.create(
        model='glm-4',
        messages=[{'role': 'user', 'content': 'Say hello in English'}],
    )
    print('SUCCESS!')
    print(f'Response: {response.choices[0].message.content}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
