import yaml

# Test loading the config
with open('knowledge_engine/config/causal_learn.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Causal-Learn Configuration Loaded Successfully')
print(f"Default algorithm: {config['general']['default_algorithm']}")
print(f"Alpha: {config['general']['alpha']}")
print(f"Available algorithms: {list(config['algorithms'].keys())}")
print(f"Output formats: {config['output']['formats']}")
