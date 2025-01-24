import yaml
class configManager:
    config = False

    def __init__(self):
        with open('data/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def get_setting(self, setting):
        return self.config[setting] or False
    
    def update_setting(self, setting, value):
        self.config[setting] = value
        with open('data/config.yaml', 'w') as file:
            yaml.dump(self.config, file)
