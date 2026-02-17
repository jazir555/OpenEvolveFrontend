"""Test Zhipu AI with available models"""
from zhipuai import ZhipuAI

client = ZhipuAI(api_key='9266cf5bd6d845f4b0e515ee3b5698a0.IouaWjdtuPNGEYyW')

# Try different models
models = ['glm-4-air', 'glm-4-flash', 'glm-4', 'glm-3-turbo', 'cogview-3']

for model in models:
    print(f'\nTrying model: {model}')
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': 'Say hello'}],
            max_tokens=20
        )
        print(f'*** SUCCESS with {model} ***')
        print(f'Response: {response.choices[0].message.content}')
        break
    except Exception as e:
        err_msg = str(e)
        if '1211' in err_msg or 'not exist' in err_msg.lower():
            print(f'  Model not available')
        else:
            print(f'  Error: {err_msg[:100]}')
