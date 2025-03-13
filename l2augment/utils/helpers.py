from l2augment.modelling import models
import torch
import pickle
import os
from os.path import join

def load_rl_models(config):
    policy_class = config.get('policy', {}).get('class', 'default') 
    policy_net = models.policy_dict[policy_class](**config.get('policy', {}).get('config', {}))
    policy_net = policy_net
    return policy_net

def lmap(func, *iterables):
    return list(map(func, *iterables))

def load_model(model, config, path=None, log_command=print):
    save_path = config.get('training', {}).get('model_save_path', None) if path == None else path
    if save_path == None:
        return 
    try:
        # Load the checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])    
        log_command(f"Model successfully loaded from {save_path}")
        return
    except FileNotFoundError:
        return 
    except Exception as e:
        log_command(f"Error loading model: {e}")
        raise

def save_model(model, config, save_path=None, log_command=print):
    save_path = config['training']['model_save_path'] if save_path is None else save_path

    isnan = False
    for _, param in model.state_dict().items():
        if torch.isnan(param).any():
            isnan = True
    if isnan == False:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, save_path)
        
        log_command(f"Model saved successfully to {save_path}")
    else:
        log_command(f"Model not saved due to NaN in weights!")


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def find_existing_run_wer(directory, id):
    files = os.listdir(directory)
    files = [el for el in files if el.split('_')[0] == str(id)]
    if len(files) > 0:
        file_pth = files[0]
        file = load_dictionary(join(directory, file_pth))
        return file['original_wer']
    return None

def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    if hasattr(asr_model, 'flash_attn'): asr_model.flash_attn = False
    return asr_model


def make_color(text, color):
    colors = { # use neon colors
        'green': '\033[38;5;46m',
        'red': '\033[38;5;196m',
        'blue': '\033[38;5;27m',
        'yellow': '\033[38;5;226m',
        'purple': '\033[38;5;129m',
        'cyan': '\033[38;5;45m',
        'white': '\033[38;5;231m',
        'orange': '\033[38;5;208m',
        'pink': '\033[38;5;198m',
        'black': '\033[38;5;0m',
    }
    assert color in colors, f"Color {color} not found. Choose from {list(colors.keys())}"
    return f"{colors[color]}{text}\033[0m"

def backward_pass(loss, model, optim):
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0) 
    optim.step()