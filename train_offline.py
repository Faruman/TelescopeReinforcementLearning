import os

from collections import deque
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from rl_model.sac_ensemble_average_diverseBatches_sameDataset import SAC as AGENT
#from rl_model.sac_ensemble_average_diverseBatches import SAC as AGENT
#from rl_model.sac_ensemble_average_equalBatches import SAC as AGENT
#from rl_model.sac_simple import SAC as AGENT
from tensorboardX import SummaryWriter

from tempfile import mkdtemp

import time

import wandb



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, replay_paths, temp_folder_path, number_of_shards, raw_input= False, data_per_shard= False):
        'Initialization'
        self.paths = []
        self.shape = {}
        self.index = np.array([0])
        if not os.path.exists(temp_folder_path):
            os.makedirs(temp_folder_path)
        self.temp_path = temp_folder_path
        self.memmaps = {"state": [], "action": [], "next_state": [], "reward": []}
        self.shard = np.array([[]])
        self.data_per_shard = data_per_shard

        self.paths.extend([replay_path for replay_path in replay_paths])
        #self.paths.extend([replay_path + "_episode_" + str(1000) for replay_path in replay_paths])
        #self.paths.extend([replay_path + "_episode_" + str(2000) for replay_path in replay_paths])

        for i, path in enumerate(self.paths):
            folder = np.load(path + ".npz")
            #print([x[0] for x in list(folder.items())])
            if raw_input:
                commands, measurements, reward, action = [x[1] for x in list(folder.items())]
                # transform shape: [experience, steps, features]
                commands = commands.reshape((int(commands.shape[0] / 1000), 1000, commands.shape[1]))
                measurements = measurements.reshape((int(measurements.shape[0] / 1000), 1000, measurements.shape[1]))
                reward = reward.reshape((int(reward.shape[0] / 1000), 1000))
                action = action.reshape((int(action.shape[0] / 1000), 1000, action.shape[1]))

                # TODO: put to zero or remove them (check again)
                commands_t1 = commands[:, 2:-1, :]
                #commands_t1 = commands_t1.reshape((commands_t1.shape[0]*commands_t1.shape[1], commands_t1.shape[2]))
                commands_t2 = commands[:, 1:-2, :]
                #commands_t2 = commands_t2.reshape((commands_t2.shape[0] * commands_t2.shape[1], commands_t2.shape[2]))
                commands_t3 = commands[:, :-3, :]
                #commands_t3 = commands_t3.reshape((commands_t3.shape[0] * commands_t3.shape[1], commands_t3.shape[2]))

                measurements = measurements[:, 3:, :]
                #measurements = measurements.reshape((measurements.shape[0] * measurements.shape[1], measurements.shape[2]))

                state = np.stack((measurements, commands_t3, commands_t2, commands_t1), axis=2).reshape(measurements.shape[0], measurements.shape[1], measurements.shape[2] + commands_t1.shape[2] + commands_t2.shape[2] + commands_t3.shape[2])
                #state = np.column_stack((measurements, commands_t1, commands_t2, commands_t3))

                # shifted according to delay (2 timesteps)
                action = action[:, :-2, :]
                action = action.reshape((action.shape[0] * action.shape[1], action.shape[2]))

                reward = reward[:,2:]
                reward = reward.reshape((reward.shape[0] * reward.shape[1]))

                next_state = state[:, 1:, :].reshape((state.shape[0] * (state.shape[1]-1), state.shape[2]))
                state = state[:, :-1, :].reshape((state.shape[0] * (state.shape[1]-1), state.shape[2]))
                folder = [("state", state), ("action", action), ("reward", reward), ("next_state", next_state)]
            else:
                folder = folder.items()

            for j, (name, item) in enumerate(folder):
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

        if raw_input:
            state_mean = np.mean(np.row_stack([np.mean(x, axis= 0) for x in self.memmaps["state"]]), axis= 0)
            state_std = np.row_stack([np.mean(np.square(x - state_mean), axis= 0) for x in self.memmaps["state"]])
            state_std = np.sqrt(np.mean(state_std, axis= 0) / state_std.shape[0])
            for i in range(len(self.memmaps["state"])):
                self.memmaps["state"][i] = (self.memmaps["state"][i] - state_mean) / state_std
                self.memmaps["next_state"][i] = (self.memmaps["next_state"][i] - state_mean) / state_std
            print(np.mean(np.row_stack([np.mean(x, axis= 0) for x in self.memmaps["state"]]), axis= 0))
            state_std = np.row_stack([np.mean(np.square(x - state_mean), axis=0) for x in self.memmaps["state"]])
            print(np.sqrt(np.mean(state_std, axis=0) / state_std.shape[0]))

        # TODO: save the standardized data

        if data_per_shard:
            self.meta_index = np.random.choice(self.index[-1], size= int(self.index[-1] * number_of_shards * self.data_per_shard), replace=True)
            self.shard = np.random.choice(number_of_shards, size= int(self.index[-1] * number_of_shards * self.data_per_shard), replace=True)
        else:
            self.shard = np.random.choice(number_of_shards, size= self.index[-1], replace=True)

  def __len__(self):
        'Denotes the total number of samples'
        if self.data_per_shard:
            return self.meta_index.shape[0]
        else:
            return self.index[-1]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select right path
        if self.data_per_shard:
            sample_idx = np.argmax(self.index > self.meta_index[index]) - 1
            subindex = self.meta_index[index] - self.index[sample_idx]
        else:
            sample_idx = np.argmax(self.index > index) - 1
            subindex = index - self.index[sample_idx]

        # Load data and get label
        state = self.memmaps["state"][sample_idx][subindex]
        action = self.memmaps["action"][sample_idx][subindex]
        next_state = self.memmaps["next_state"][sample_idx][subindex]
        reward = np.array(self.memmaps["reward"][sample_idx][subindex])
        shard = self.shard[index]
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
        self.sac['shuffle'] = True
        self.sac['updates_per_step'] = 1

        self.cuda = True

        self.env_rl = dict()
        self.env_rl['write_every'] = 1
        self.env_rl["save_path"] = "."
        self.env_rl['seed'] = 42


class TrainerOffline:
    """
    Class that manages everything related to training of the AO RL agent
    """
    def __init__(self,
                 replay_paths,
                 config,
                 experiment_name,
                 writer,
                 env_name,
                 raw_input = False,
                 data_per_shard= False,
                 reload_old = False
                 ):
        """
        replay_paths: list of replay paths
        config: sac config
        experiment_name: name of experiment
        writer: tensorboardX writer
        """
        self.env_name = env_name

        train_dataset = Dataset(replay_paths, "./temp/", config.sac['size_ensemble'], raw_input= raw_input, data_per_shard= data_per_shard)

        self.agent = AGENT(num_inputs= train_dataset.shape["state"],
                           num_actions=  train_dataset.shape["action"],
                           config=config)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=config.sac['batch_size'],
                                                        shuffle=config.sac['shuffle'])

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

        if reload_old:
            self.epoch = self.agent.load_model(self.env_name)

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


exp_name = "10x10_rl"
writ = SummaryWriter(exp_name)
conf = Config()

torch.manual_seed(conf.env_rl["seed"])
random.seed(conf.env_rl["seed"])
np.random.seed(conf.env_rl["seed"])

# train model on data from the simple agent
base_path = r"./data/"
rep_paths = [
            r"replay_11_2m_fabian_noise_mu_0_sigma_05_episode_2000-001",
            r"replay_11_2m_fabian_noise_mu_0_sigma_01_episode_2000-002",
            r"replay_11_2m_fabian_noise_mu_0_sigma_03_episode_2000-003"
            ]
rep_paths = [base_path + rep_path for rep_path in rep_paths]

trainer = TrainerOffline(replay_paths=rep_paths,
                         experiment_name=exp_name,
                         config=conf,
                         writer=writ,
                         env_name= "sac_ensemble_2m_{}".format(exp_name),
                         raw_input= False,
                         data_per_shard = False,
                         reload_old = True
                         )

trainer.train_agent(num_epochs= 300)
wandb.finish()