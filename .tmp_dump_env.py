import os, json\nwith open('.tmp_env_dump.txt','w',encoding='utf-8') as f:\n    f.write(json.dumps({k:v for k,v in os.environ.items() if k.startswith('API_KEY_')}, indent=2))\n
