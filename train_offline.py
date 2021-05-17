import os

from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
#from rl_model.sac_ensemble_average_diverseBatches_sameDataset import SAC as AGENT
from rl_model.sac_ensemble_average_diverseBatches import SAC as AGENT
#from rl_model.sac_ensemble_average_equalBatches import SAC as AGENT
from tensorboardX import SummaryWriter

from tempfile import mkdtemp

import time

import wandb



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, replay_paths, temp_folder_path, number_of_shards):
        'Initialization'
        self.paths = []
        self.shape = {}
        self.index = np.array([0])
        if not os.path.exists(temp_folder_path):
            os.makedirs(temp_folder_path)
        self.temp_path = temp_folder_path
        self.memmaps = {"state": [], "action": [], "next_state": [], "reward": [], "shard": []}

        self.paths.extend([replay_path + "_episode_" + str(1000) for replay_path in replay_paths])
        self.paths.extend([replay_path + "_episode_" + str(2000) for replay_path in replay_paths])

        for i, path in enumerate(self.paths):
            folder = np.load(path + ".npz")
            for j, (name, item) in enumerate(folder.items()):
                self.memmaps[name].append([])
                self.memmaps[name][i] = np.memmap(os.path.join(self.temp_path, os.path.basename(path)) + "_" + name + ".dat", dtype='float32', mode='w+', shape=item.shape)
                self.memmaps[name][i][:] = item
                self.memmaps[name][i].flush()
                if j == 0:
                    self.index = np.append(self.index, item.shape[0] + self.index[-1])
                if len(item.shape) > 1:
                    self.shape[name] = item.shape[1]
                else:
                    self.shape[name] = 1
            self.memmaps["shard"].append([])
            self.memmaps["shard"][i] = np.memmap(os.path.join(self.temp_path, os.path.basename(path)) + "_shard.dat", dtype='float32', mode='w+', shape=(self.index[i + 1], 1))
            self.memmaps["shard"][i][:] = np.expand_dims(np.random.choice(number_of_shards, self.index[i + 1]), 1)
            self.memmaps["shard"][i].flush()

  def __len__(self):
        'Denotes the total number of samples'
        return self.index[-1]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select right path

        sample_idx = np.argmax(self.index > index) - 1
        index = index - self.index[sample_idx]

        # Load data and get label
        state = self.memmaps["state"][sample_idx][index]
        action = self.memmaps["action"][sample_idx][index]
        next_state = self.memmaps["next_state"][sample_idx][index]
        reward = np.array(self.memmaps["reward"][sample_idx][index])
        shard = np.array(self.memmaps["shard"][sample_idx][index])
        return state, action, next_state, reward, shard


class Config:
    """
    Class for config of SAC
    """
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

        self.env_rl["save_path"] = "."


class TrainerOffline:
    """
    Class that manages everything related to training of the AO RL agent
    """
    def __init__(self,
                 replay_paths,
                 config,
                 experiment_name,
                 writer,
                 env_name
                 ):
        """
        replay_paths: list of replay paths
        config: sac config
        experiment_name: name of experiment
        writer: tensorboardX writer
        """
        self.env_name = env_name

        train_dataset = Dataset(replay_paths, "./temp/", config.sac['size_ensemble'])

        self.agent = AGENT(num_inputs= train_dataset.shape["state"],
                           num_actions=  train_dataset.shape["action"],
                           config=config)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=config.sac['batch_size'],
                                                        shuffle=False)

        self.writer = writer
        self.config, self.experiment_name = config, experiment_name
        self.len_actions = train_dataset.shape["action"]
        self.epoch = 0

        self.sac_info_dic = {"critic_1_loss": [],
                             "critic_2_loss": [],
                             "policy_loss": [],
                             "ent_loss": [],
                             "alpha": []}

        wandb.init(project='capstoneProject', entity='faruman')
        self.total_update = 0

    def train_agent(self, num_epochs):
        """
        Function that manages the full training loop of the agent
        """

        print("SAC used for training offline")

        while self.epoch < num_epochs:
            self.episode()

            if self.epoch % 3 == 0 and self.epoch != 0:
                self.agent.save_model(self.env_name, episode= self.epoch)

        self.agent.save_model(self.env_name)

    def episode(self):
        """
        Does an episode of the environment
        The episode definition depends on the configuration
        """
        print("Epoch:", self.epoch)
        time.sleep(0.5)

        self.update_sac_offline()
        self.total_update += 2

        self.epoch += 1

    def update_sac_offline(self):
        """
        Updates SAC
        """

        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
            self.agent.update_parameters_sac_offline(self.train_loader, num_epochs=1, env_name = self.env_name)
        self.sac_info_dic['critic_1_loss'].append(critic_1_loss)
        self.sac_info_dic['critic_2_loss'].append(critic_2_loss)
        self.sac_info_dic['policy_loss'].append(policy_loss)
        self.sac_info_dic['ent_loss'].append(ent_loss)
        self.sac_info_dic['alpha'].append(alpha)

        if self.total_update % self.config.env_rl['write_every'] == 0:
            self.writer.add_scalar("sac/critic_1_loss", np.mean(self.sac_info_dic['critic_1_loss']),
                                   self.total_update * self.config.sac['updates_per_step'])
            self.writer.add_scalar("sac/critic_2_loss", np.mean(self.sac_info_dic['critic_2_loss']),
                                   self.total_update * self.config.sac['updates_per_step'])
            self.writer.add_scalar("sac/policy_loss", np.mean(self.sac_info_dic['policy_loss']),
                                   self.total_update * self.config.sac['updates_per_step'])
            self.writer.add_scalar("sac/ent_loss", np.mean(self.sac_info_dic['ent_loss']),
                                   self.total_update * self.config.sac['updates_per_step'])
            self.writer.add_scalar("sac/alpha", np.mean(self.sac_info_dic['alpha']),
                                   self.total_update * self.config.sac['updates_per_step'])

            self.sac_info_dic['critic_1_loss'] = []
            self.sac_info_dic['critic_2_loss'] = []
            self.sac_info_dic['policy_loss'] = []
            self.sac_info_dic['ent_loss'] = []
            self.sac_info_dic['alpha'] = []


exp_name = "test"
writ = SummaryWriter(exp_name)
conf = Config()


# train model on data from the simple agent
rep_paths = [r"data/replay_2m_10x10_linear"]
trainer = TrainerOffline(replay_paths=rep_paths,
                         experiment_name=exp_name,
                         config=conf,
                         writer=writ,
                         env_name= "sac_ensemble_2m_10x10_linear")
trainer.train_agent(num_epochs= 150)
wandb.finish()