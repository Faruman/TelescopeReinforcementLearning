import torch
from glob import glob
from rl_model import model_simple, sac_ensemble, sac_ensemble_variable

class Config:
    def __init__(self):
        self.sac = dict()
        self.sac['sac_reward_scaling'] = 1.0
        self.sac['gamma'] = 0.5
        self.sac['tau'] = 0.005
        self.sac['alpha'] = 0.2

        self.sac['target_update_interval'] = 1
        self.sac['automatic_entropy_tuning'] = True

        self.sac['initialize_last_layer_0'] = True
        self.sac['initialize_last_layer_near_0'] = False

        self.sac['activation'] = "relu"

        self.sac['lr'] = 0.0003
        self.sac['hidden_size_critic'] = 256
        self.sac['num_layers_critic'] = 2
        self.sac['hidden_size_actor'] = 256
        self.sac['num_layers_actor'] = 2
        self.sac['size_ensemble'] = 5
        self.sac['batch_size'] = 256

        self.sac['updates_per_step'] = 1

        self.cuda = True

        self.env_rl = dict()
        self.env_rl['write_every'] = 1

config = Config()

reward_scale = config.sac['sac_reward_scaling']
gamma = config.sac['gamma']
tau = config.sac['tau']
alpha = config.sac['alpha']

target_update_interval = config.sac['target_update_interval']
automatic_entropy_tuning = config.sac['automatic_entropy_tuning']

initialize_last_layer_zero = config.sac['initialize_last_layer_0']
initialize_last_layer_near_zero = config.sac['initialize_last_layer_near_0']

activation = config.sac['activation']

lr = config.sac['lr']
hidden_size_critic = config.sac['hidden_size_critic']
num_layers_critic = config.sac['num_layers_critic']
hidden_size_actor = config.sac['hidden_size_actor']
num_layers_actor = config.sac['num_layers_actor']

num_inputs = 359
num_actions = 77

model_paths = glob(r"D:\Programming\Python\CapstoneProject\output\models\sac_ensemble_average\*", recursive= False)

for model_path in model_paths:
    model = None
    if "actor" in model_path:
        model = sac_ensemble_variable.GaussianPolicy(num_inputs=num_inputs,
                                         num_actions=num_actions,
                                         hidden_dim=hidden_size_actor,
                                         action_scale=1.0,
                                         action_bias=0.0,
                                         num_layers=num_layers_actor,
                                         initialize_last_layer_zero=initialize_last_layer_zero,
                                         initialize_last_layer_near_zero=initialize_last_layer_near_zero,
                                         activation=activation
                                         )
    elif "critic" in model_path:
        model = sac_ensemble_variable.QNetwork(num_inputs,
                                      num_actions,
                                      hidden_size_critic,
                                      num_layers_critic)

    if "q_transform" not in model_path and model:
        model.load_state_dict(torch.load(model_path))
        torch.save(model.state_dict(), model_path + "_old", _use_new_zipfile_serialization=False)