from litellm import completion

key = '9555ab1f79be4db498d0cf19c1277af0.8lf475NPrUgEWaF0'

print('Testing Zhipu AI via LiteLLM...')

# Try different model formats - glm-5 is the latest
models_to_try = [
    'zhipuai/glm-5',
    'zhipuai/glm-4',
    'zhipuai/glm-4-0520',
    'zhipuai/chatglm_turbo',
]

for model in models_to_try:
    print(f'\nTrying model: {model}')
    try:
        response = completion(
            model=model,
            messages=[{'role': 'user', 'content': 'Say hello in English'}],
            api_key=key,
            max_tokens=20,
            timeout=30
        )
        print('SUCCESS!')
        print('Response:', response.choices[0].message.content)
        break
    except Exception as e:
        print('Error:', str(e)[:200])
else:
    print('\nAll models failed')
