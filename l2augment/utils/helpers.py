from l2augment.modelling import models as policy_models

def load_rl_models(config):
    policy_class = config.get('policy', {}).get('class', 'default') 
    policy_net = policy_models.policy_dict[policy_class](**config.get('policy', {}).get('config', {}))
    policy_net = policy_net
    return policy_net