def get_config():
    import os
    import yaml
    config_path = os.path.join(os.getcwd(), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config