import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model_simple import GaussianPolicy, QNetwork
import numpy as np
import collections
import matplotlib.pyplot as plt

from tqdm import tqdm
import wandb


class SAC(object):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 config
                 ):

        self.config = config

        self.reward_scale = config.sac['sac_reward_scaling']
        self.gamma = config.sac['gamma']
        self.tau = config.sac['tau']
        self.alpha = config.sac['alpha']

        self.target_update_interval = config.sac['target_update_interval']
        self.automatic_entropy_tuning = config.sac['automatic_entropy_tuning']

        initialize_last_layer_zero = config.sac['initialize_last_layer_0']
        initialize_last_layer_near_zero = config.sac['initialize_last_layer_near_0']

        activation = config.sac['activation']

        self.lr = config.sac['lr']
        hidden_size_critic = config.sac['hidden_size_critic']
        num_layers_critic = config.sac['num_layers_critic']
        hidden_size_actor = config.sac['hidden_size_actor']
        num_layers_actor = config.sac['num_layers_actor']

        self.size_ensemble = config.sac['size_ensemble']

        self.device = torch.device("cuda" if config.cuda else "cpu")

        self.critic = list()
        self.critic_target = list()
        self.critic_optim = list()
        for i in range(self.size_ensemble):
            self.critic.append(QNetwork(num_inputs, num_actions, hidden_size_critic, num_layers_critic).to(
                device=self.device))
            self.critic_target.append(QNetwork(num_inputs, num_actions, hidden_size_critic, num_layers_critic).to(
                self.device))
            hard_update(self.critic_target[-1], self.critic[-1])
            self.critic_optim.append(Adam(self.critic[i].parameters(), lr=self.lr))
        self.critic_diff = np.zeros(shape=(0, 2))

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(num_actions).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
            self.policy = GaussianPolicy(num_inputs=num_inputs,
                                         num_actions=num_actions,
                                         hidden_dim=hidden_size_actor,
                                         action_scale=1.0,
                                         action_bias=0.0,
                                         num_layers=num_layers_actor,
                                         initialize_last_layer_zero=initialize_last_layer_zero,
                                         initialize_last_layer_near_zero=initialize_last_layer_near_zero,
                                         activation=activation
                                         ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        # TODO delete
        self.actor_activated = True
        self.sac_writing_critic=0


    def select_action(self, state, evalu=False, return_log_pi=False):
        """
        Selects action based on current policy
        :param state: s
        :param evalu: if evaluation we do not sample and choose deterministically
        :param return_log_pi: if we return logπ
        :return:
        """

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evalu is False:
            action, logpi, _ = self.policy.sample(state)
        else:
            _, logpi, action = self.policy.sample(state)

        action_to_return = action.detach().cpu().numpy()[0]

        if return_log_pi:
            return action_to_return, logpi.detach().cpu().numpy().item()
        else:
            return action_to_return

    # --SAC UPDATE--

    def update_critic(self,
                      state_batch,
                      action_batch,
                      reward_batch,
                      next_state_batch,
                      mask_batch):
        """
        Updates Q(s,a) based on Bellman eq.
        :param state_batch: batch of s
        :param action_batch: batch of a
        :param reward_batch: batch of r
        :param next_state_batch: batch of s'
        :param mask_batch: batch of not dones
        :return: items for tensorboardX
        """

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)

        qf1_loss_values = torch.empty(self.size_ensemble).to(self.device)
        qf2_loss_values = torch.empty(self.size_ensemble).to(self.device)

        for i in range(self.size_ensemble):
            with torch.no_grad():
                qf1_next_target, qf2_next_target = self.critic_target[i](next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

            qf1, qf2 = self.critic[i](state_batch, action_batch)
            # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            qf_loss = qf1_loss + qf2_loss

            self.critic_optim[i].zero_grad()
            qf_loss.backward()
            self.critic_optim[i].step()

            qf1_loss_values[i] = qf1_loss.detach()
            qf2_loss_values[i] = qf2_loss.detach()

        qf1_loss_combined = torch.mean(qf1_loss_values)
        qf2_loss_combined = torch.mean(qf2_loss_values)

        return qf1_loss_combined.item(), qf2_loss_combined.item()

    def update_actor(self, state_batch):
        """
        Updates π
        :param state_batch: batch of states
        :return: items for tensorboardX
        """
        pi, log_pi, _ = self.policy.sample(state_batch)

        min_qf_pi = torch.empty(self.size_ensemble, state_batch.shape[0], 1).to(self.device)

        for i in range(self.size_ensemble):
            qf1_pi, qf2_pi = self.critic[i](state_batch, pi)
            min_qf_pi[i] = torch.min(qf1_pi, qf2_pi)
            # TODO: Maybe add a debug flag
            # Check if the critic give a similiar value
            if np.random.random() < 0.001:
                critics_exmp = min_qf_pi[:, 0].detach().cpu().numpy().flatten()
                print("Critics: {}".format(critics_exmp))
                self.critic_diff = np.append(self.critic_diff, np.array([[np.min(critics_exmp), np.max(critics_exmp)]]), axis=0)

        min_qf_pi =  torch.mean(torch.transpose(min_qf_pi[:, :, 0], 0, 1))



        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return log_pi, policy_loss.item()

    def update_alpha(self, log_pi):
        """
        Updates alpha if entropy tunning is activated
        :param log_pi: logπ
        :return: alpha items for tensorboard
        """
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs

        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return alpha_loss.item(), alpha_tlogs.item()

    def update_parameters_sac_offline(self,
                                      memory,
                                      num_epochs=2,
                                      env_name = ""
                                      ):
        """
        Updates SAC from a dataset
        memory: train loader
        number of epochs: number of epochs. This is not used anymore but I was looking into the simulation performance
        after this number of epochs
        """

        loss_qf1_list = []
        loss_qf2_list = []
        loss_policy_loss_list = []
        alpha_loss_value_list = []
        alpha_tlogs_value_list = []

        for epoch in range(num_epochs):
            for i, (state_batch, action_batch, next_state_batch, reward_batch) in enumerate(tqdm(memory, desc= "Train")):
                mask_batch = torch.ones_like(reward_batch).to(self.device)

                state_batch, action_batch, next_state_batch, reward_batch =\
                    state_batch.to(self.device), action_batch.to(self.device),\
                    next_state_batch.to(self.device), reward_batch.to(self.device)

                qf1_loss_value, qf2_loss_value = self.update_critic(state_batch,
                                                                    action_batch,
                                                                    reward_batch,
                                                                    next_state_batch,
                                                                    mask_batch)

                log_pi, policy_loss_value = self.update_actor(state_batch)

                alpha_loss_value, alpha_tlogs_value = self.update_alpha(log_pi)

                for i in range(self.size_ensemble):
                    soft_update(self.critic_target[i], self.critic[i], self.tau)

                loss_qf1_list.append(qf1_loss_value)
                loss_qf2_list.append(qf2_loss_value)
                loss_policy_loss_list.append(policy_loss_value)
                alpha_loss_value_list.append(alpha_loss_value)
                alpha_tlogs_value_list.append(alpha_tlogs_value)

                # TODO: Maybe add a default flag
                if i % 100 == 0:
                    print("critic_1_loss: {}, critic_2_loss: {}, policy_loss: {}, ent_loss: {}, alpha: {}".format(
                        qf1_loss_value, qf2_loss_value, policy_loss_value, alpha_loss_value, alpha_tlogs_value))

                wandb.log({"critic_1_loss": qf1_loss_value,
                           "critic_2_loss": qf2_loss_value,
                           "policy_loss": policy_loss_value,
                           "ent_loss": alpha_loss_value,
                           "alpha": alpha_tlogs_value})

            plt.plot(self.critic_diff[:, 0])
            plt.plot(self.critic_diff[:, 1])
            plt.title("min/max prediction critics")
            plt.savefig("./plots/critics_diff_{}.png".format(env_name))
            plt.show()

        return np.mean(loss_qf1_list),\
            np.mean(loss_qf2_list),\
            np.mean(loss_policy_loss_list),\
            np.mean(alpha_loss_value_list),\
            np.mean(alpha_tlogs_value_list)

    # Save model parameters
    def save_model(self,
                   env_name,
                   episode=None,
                   actor_path=None,
                   critic_path=None):
        """
        TODO
        :param env_name:
        :param episode:
        :param actor_path:
        :param critic_path:
        :return:
        """
        if not os.path.exists('./output/models/'):
            os.makedirs('./output/models/')

        if actor_path is None:
            actor_path = "./output/models/sac_actor_{}".format(env_name)
        if critic_path is None:
            critic_path = "./output/models/sac_critic_{}".format(env_name)

        if episode is not None:
            actor_path += "_episode_" + str(episode)
            critic_path += "_episode_" + str(episode)

        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path, _use_new_zipfile_serialization=False)
        for i, ind_critic in enumerate(self.critic):
            torch.save(ind_critic.state_dict(), critic_path + "_" + str(i), _use_new_zipfile_serialization=False)

    # Load model parameters
    def load_model(self,
                   actor_path,
                   critic_paths,
                   critic_q_transform_path):
        """
        :param actor_path:
        :param critic_path:
        :return:
        """
        print('Loading models from {} and {}'.format(actor_path, critic_paths))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))

        if critic_paths is not None:
            for i, critic_path in enumerate(critic_paths):
                self.critic[i].load_state_dict(torch.load(critic_path))
