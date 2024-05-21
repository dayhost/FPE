"""
本文件用于将Environment和Agent两个模块串联起来并且训练Agent
"""
import datetime
import random
import gym
import pickle
import torch.nn.functional as functional
import os
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (6, 5)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
import matplotlib.pylab as pl

# setting use of the GPU or CPU
USE_CUDA = torch.cuda.is_available()
# if the GPU is available for the server, the device is GPU, otherwise, the device is CPU
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


def plot_policies(policy_list, true_loss_list, semi_loss_list, state_size, file_save_path):
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    total_policy_pd = pd.DataFrame()
    for i in range(len(policy_list)):
        policy_dict = {"iter" + str(i): {}}
        for j in range(len(policy_list[i])):
            policy_dict["iter" + str(i)][r"$s_{" + str(j+1) + "}$"] = policy_list[i][j]
        total_policy_pd = pd.concat([total_policy_pd, pd.DataFrame().from_dict(policy_dict)], axis=1)

    total_policy_pd = total_policy_pd.sort_index(key=lambda key_list: [int(tmp.split("{")[1].split("}")[0]) for tmp in key_list])

    myColors = ((255/255, 247/255, 234/255, 1), (252/255, 192/255, 142/255, 1), (231/255, 86/255, 59/255, 1), (106/255, 7/255, 10/255, 1))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    ax = sns.heatmap(total_policy_pd, cmap=cmap)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.4, 1.15, 1.9, 2.6])
    colorbar.set_ticklabels([r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'])

    x_label = list(np.arange(0, len(policy_list)))

    ax.set_xticks(range(len(x_label)))
    ax.set_xticklabels(x_label)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='y', labelrotation=0)

    for state in range(state_size):
        plt.plot(x_label, [state+1] * len(x_label), color="white")

    plt.axvline(mid_idx, color="white", linestyle="--")
    plt.axvline(max_after_mid_idx, color="white", linestyle="--")
    plt.text(x=mid_idx - 2000, y=9, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")
    plt.text(x=max_after_mid_idx - 2000, y=9, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")

    plt.xlabel("training step")
    plt.ylabel("state")
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_grid_world_dqn_policy_change.png")
    plt.close()


def plot_partial_policies(policy_list, true_loss_list, semi_loss_list, state_size, start_idx, end_idx, file_save_path):
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    partial_policy_list = policy_list[start_idx:end_idx]

    total_policy_pd = pd.DataFrame()
    for i in range(len(partial_policy_list)):
        policy_dict = {"iter" + str(i + start_idx): {}}
        for j in range(len(partial_policy_list[i])):
            policy_dict["iter" + str(i + start_idx)][r"$s_{" + str(j + 1) + "}$"] = partial_policy_list[i][j]
        total_policy_pd = pd.concat([total_policy_pd, pd.DataFrame().from_dict(policy_dict)], axis=1)

    total_policy_pd = total_policy_pd.sort_index(
        key=lambda key_list: [int(tmp.split("{")[1].split("}")[0]) for tmp in key_list])

    myColors = (
    (255 / 255, 247 / 255, 234 / 255, 1), (252 / 255, 192 / 255, 142 / 255, 1), (231 / 255, 86 / 255, 59 / 255, 1),
    (106 / 255, 7 / 255, 10 / 255, 1))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    ax = sns.heatmap(total_policy_pd, cmap=cmap)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.4, 1.15, 1.9, 2.6])
    colorbar.set_ticklabels([r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'])

    x_label = list(range(start_idx, start_idx + len(partial_policy_list)))

    ax.set_xticks(range(len(x_label)))
    ax.set_xticklabels(x_label)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='y', labelrotation=0)

    for state in range(state_size):
        plt.plot(x_label, [state + 1] * len(x_label), color="white")

    if start_idx <= mid_idx <= end_idx:
        plt.axvline(mid_idx - start_idx, color="white", linestyle="--")
        plt.text(x=mid_idx - start_idx + 50, y=7, s=r"$t_1$",
                 fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")

    if start_idx <= max_after_mid_idx <= end_idx:
        plt.axvline(max_after_mid_idx - start_idx, color="white", linestyle="--")
        plt.text(x=max_after_mid_idx - start_idx + 50, y=7, s=r"$t_2$",
                 fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")

    plt.xlabel("training step")
    plt.ylabel("state")
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_grid_world_dqn_policy_change.png")
    plt.close()


def plot_loss(true_loss_list, semi_loss_list, file_save_path):

    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    print("max_after_mid_idx: " + str(max_after_mid_idx))

    plt.plot(np.arange(0, len(true_loss_list), 1), true_loss_list, color="red")
    plt.plot(np.arange(len(true_loss_list), len(true_loss_list) + len(semi_loss_list), 1), semi_loss_list, color="blue")

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx - 2000, y=-2, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx - 2000, y=-2, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    plt.scatter(max_after_mid_idx, max(semi_loss_list), marker="X", s=200, color="orange")

    plt.xlabel("training step")
    plt.ylabel("loss")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_grid_world_dqn_loss.png")
    plt.close()


def plot_log_loss(true_loss_list, semi_loss_list, file_save_path):

    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    print("max_after_mid_idx: " + str(max_after_mid_idx))

    plt.semilogy(np.arange(0, len(true_loss_list), 1), true_loss_list, color="red")
    plt.semilogy(np.arange(len(true_loss_list), len(true_loss_list) + len(semi_loss_list), 1),
                 semi_loss_list, color="blue")

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx - 2000, y=1e-2, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx - 2000, y=1e-2, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    # plt.scatter(max_after_mid_idx, max(semi_loss_list), marker="X", s=200, color="orange")

    plt.xlabel("training step")
    plt.ylabel("loss")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_grid_world_dqn_loss.png")
    plt.close()


def plot_value_function(true_state_value_tuple_list, semi_state_value_tuple_list, true_loss_list, semi_loss_list,
                        state_size, file_save_path):
    colors = pl.cm.jet(np.linspace(0, 1, state_size))

    line_list = []
    label_list = []

    for state in range(state_size):

        label_list.append(r"$s_{" + str(state+1) + "}$")

        # 画true的线
        data_dict_list = []
        for i in range(len(true_state_value_tuple_list)):
            data_dict_list.append(true_state_value_tuple_list[i][state])
        l1, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list,
                       label=r"$s_{" + str(state+1) + "}$", color=colors[state])

        # 开始画State Value的Semi State的线
        data_dict_list = []
        for i in range(len(semi_state_value_tuple_list)):
            data_dict_list.append(semi_state_value_tuple_list[i][state])
        l2, = plt.plot(list(np.arange(len(true_state_value_tuple_list),
                       len(true_state_value_tuple_list) + len(semi_state_value_tuple_list),1)),
                       data_dict_list, color=colors[state], label=r"$s_{" + str(state+1) + "}$", linestyle="--")

        line_list.append((l1, l2))

    plt.legend(handles=line_list, labels=label_list, handler_map={tuple: HandlerTuple(ndivide=None)},
               ncol=int(state_size / 5), fontsize=14)

    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx - 2000, y=0.1, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx - 2000, y=0.1, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

    plt.xlabel("training step")
    plt.ylabel("state value")
    plt.tight_layout(pad=0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    plt.savefig(file_save_path + "/true_semi_value_function_for_each_state_grid_world_dqn.png")
    plt.close()


class DeterministicMDP(gym.Env):
    """
    这个环节的表示为3x5的格子，最下一行中间的三个格子都是负奖励，左下角为出发点，右下角为终点。其余部分都是空白的格子
    """

    # 我们在初始函数中定义一个 viewer ，即画板
    def __init__(self, state_representation_size, max_step, save_file_path):
        super(DeterministicMDP, self).__init__()
        self.current_state = 10
        self.action_size = 4
        self.state_size = 19
        self.state_representation_size = state_representation_size
        self.viewer = None

        self.max_step = max_step
        self.step_count = 0

        # action的顺序为上下左右
        self.transition_matrix = self._generate_transition_tensor()
        self.transition_matrix = np.array(self.transition_matrix)

        self.step_reward = -0.01
        self.trap_penalty = -1
        self.good_reward = 1
        self.reward_tensor = self._generate_reward_tensor()
        self.reward_tensor = np.array(self.reward_tensor)

        # 读取或者创建新的MDP模型
        model_file_path = save_file_path + "/deter_mdp_" + str(self.state_size) + "_" \
                          + str(self.action_size) + "_" + str(self.state_representation_size) + ".pl"
        if os.path.exists(model_file_path):
            self.state_representation = pickle.load(open(model_file_path, "rb"))
            print("load mdp model!")
        else:
            # 构造状态的dense表示和状态和表示向量之间的映射
            full_rank = False
            while not full_rank:
                self.state_representation = np.random.rand(self.state_size, self.state_representation_size)

                if np.linalg.matrix_rank(self.state_representation) == min(self.state_size, self.state_representation_size):
                    full_rank = True

            # 每一个行向量表示一个状态，让每一个状态的L2 norm都为1
            for i in range(self.state_representation.shape[0]):
                self.state_representation[i] /= np.linalg.norm(self.state_representation[i], ord=2)

            pickle.dump(self.state_representation, open(model_file_path, "wb"))
            print("create new mdp model and dump!")

        # 构造从状态的encoding到状态下标的映射
        self.state_encode_to_index_mapping = {}
        for i in range(self.state_size):
            self.state_encode_to_index_mapping[str(self.state_representation[i].tolist())] = i

    @staticmethod
    def _transition_rule(state, action):
        """
        传入当前状态和动作，给出下一个状态
        :param state:
        :param action:
        :return:
        """
        if action == 0:
            if state < 5:
                return state
            elif 5 <= state <= 14:
                return state - 5
            else:
                return state - 4
        elif action == 1:
            if state > 15 or state == 10:
                return state
            elif 11 <= state <= 14:
                return state + 4
            else:
                return state + 5
        elif action == 2:
            if state in [0, 5, 10]:
                return state
            else:
                return state - 1
        elif action == 3:
            if state in [4, 9, 14, 18]:
                return state
            else:
                return state + 1
        else:
            raise Exception("")

    @staticmethod
    def _reward_rule(next_state, trap_reward, step_reward, good_reward):
        if next_state in [11, 12, 13]:
            return trap_reward
        elif next_state == 14:
            return good_reward
        else:
            return step_reward

    @staticmethod
    def _generate_one_hot_array(total_length, pos, value):
        one_hot_array = [0] * total_length
        one_hot_array[pos] = value
        return one_hot_array

    def _generate_transition_tensor(self):
        # 根据移动规则，形成转移概率
        transition_tensor = []
        for action in range(self.action_size):
            transition_tensor.append([])
            for state in range(self.state_size):

                if state in self.get_termination_state():
                    continue

                next_state = DeterministicMDP._transition_rule(state, action)
                transition_tensor[action].append(DeterministicMDP._generate_one_hot_array(self.state_size, next_state, 1))
        return transition_tensor

    def _generate_reward_tensor(self):
        reward_tensor = []
        for action in range(self.action_size):
            reward_tensor.append([])
            for state in range(self.state_size):

                if state in self.get_termination_state():
                    continue

                next_state = DeterministicMDP._transition_rule(state, action)
                next_reward = DeterministicMDP._reward_rule(next_state=next_state, trap_reward=self.trap_penalty,
                                                            step_reward=self.step_reward, good_reward=self.good_reward)
                reward_tensor[action].append(DeterministicMDP._generate_one_hot_array(self.state_size, next_state, next_reward))
        return reward_tensor

    def get_termination_state(self):
        return [15]

    def reset(self, state=None):
        if state is None:
            self.current_state = 10
        else:
            self.current_state = state
        self.step_count = 0

    def get_index_by_obs(self, obs):
        return self.state_encode_to_index_mapping[str(list(obs))]

    def close(self):
        # self.viewer.close()
        pass

    def obs(self):
        return self.state_representation[self.current_state]

    def step(self, action):

        if int(action) >= self.action_size:
            raise Exception("")

        next_state = np.argmax(self.transition_matrix[action][self.current_state])

        reward = self.reward_tensor[action][self.current_state][next_state]

        if next_state in self.get_termination_state() or self.step_count >= self.max_step:
            done = True
        else:
            done = False

        self.current_state = next_state
        self.step_count += 1

        return self.current_state, reward, done

    def get_full_state_action_state_state_tuple(self):

        sas_tuple_list = []

        for current_state in range(self.state_size):

            self.reset(current_state)

            for action in range(self.action_size):
                tmp_sas_tuple = [self.obs(), action]
                obs, reward, done = self.step(action)
                tmp_sas_tuple.append(self.obs())
                tmp_sas_tuple.append(reward)
                tmp_sas_tuple.append(done)
                sas_tuple_list.append(tmp_sas_tuple)

                self.current_state = current_state

        return sas_tuple_list

    def get_partial_data_tuple(self, partial_state_size):

        sas_tuple_list = []

        for current_state in range(partial_state_size):

            self.reset(current_state)

            for action in range(self.action_size):
                tmp_sas_tuple = [self.obs(), action]
                obs, reward, done = self.step(action)
                tmp_sas_tuple.append(self.obs())
                tmp_sas_tuple.append(reward)
                tmp_sas_tuple.append(done)
                sas_tuple_list.append(tmp_sas_tuple)

                self.current_state = current_state

        return sas_tuple_list


class SimpleReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_tuple = []
        self.buffer_size = buffer_size
        self.data_num_count = 0

    def insert_data_tuple(self, current_state, current_action, next_state, reward):
        """
        本函数采用循环链表插入的方法，即当Buffer满了之后，会从index0处开始进行数据覆盖
        :param current_state:
        :param current_action:
        :param next_state:
        :param reward:
        :return:
        """
        if self.get_buffer_length() < self.buffer_size:
            self.buffer_tuple.append((current_state, current_action, next_state, reward))
        else:
            self.buffer_tuple[self.data_num_count % self.buffer_size] = \
                (current_state, current_action, next_state, reward)
        self.data_num_count += 1

    def insert_data_tuple_list(self, data_tuple_list):
        for data_tuple in data_tuple_list:
            self.insert_data_tuple(data_tuple["current_state"], data_tuple["current_action"], data_tuple["next_state"],
                                   data_tuple["reward"])

    def get_sequential_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if self.data_num_count < batch_size:
            batch_size = self.data_num_count

        return self.buffer_tuple[0: batch_size]

    def get_shuffle_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if batch_size < len(self.buffer_tuple):
            result = random.sample(self.buffer_tuple, batch_size)
        else:
            result = random.sample(self.buffer_tuple, len(self.buffer_tuple))

        return result

    def get_buffer_length(self):
        return len(self.buffer_tuple)



class FCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_size)

    def forward(self, data):
        x = functional.relu(self.fc1(data))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def training_fcnn_model_with_semi_gradient(fcnn_model, optimizer, input_data, device, gamma):
    """
    本函数用于针对一个模型，一次训练数据，进行一次训练

    本函数使用常用的Semi gradient的方式进行训练。

    :param input_data: 是一个List of Tuple，每个Tuple里面包含的内容有（s_t, a_t, s_t+1, r_t），其维度分别为为
        s_t: (160, 160, 3) a_t : (3, 1) s_t+1: (160, 160, 3) r_t: (1, )
    :param simple_cnn_model: 需要训练的模型，已经放到Device上了。这里device指CPU或者GPU。
    :param optimizer: 当前模型的优化器，需要是全局性的，因为里面可能会包含类似Momentum之类的全局信息
    :param device:
    :param gamma: discounted factor
    :return:
    """
    # 将前一次Tensor中的grad清零，以便于后续的运算
    optimizer.zero_grad()

    # 将model复制一份,用于之后的计算
    duplicated_model = FCNN(input_size=fcnn_model.input_size, output_size=fcnn_model.output_size)
    duplicated_model.load_state_dict(fcnn_model.state_dict())
    duplicated_model.to(device)

    # 获取所有的s_t数据，处理后放到device上
    current_state_list = []
    for data_tuple in input_data:
        current_state_list.append(data_tuple[0])
    current_state_tensor = Variable(torch.tensor(current_state_list, device=device, dtype=torch.float32))

    # 获取所有的s_t+1数据，并挑选出那些不为final state的状态作为函数的输入
    non_final_next_state_list = []
    next_state_list = []
    for data_tuple in input_data:
        next_state_list.append(data_tuple[2])
        if data_tuple[2] is not None:
            non_final_next_state_list.append(data_tuple[2])

    if len(non_final_next_state_list) == 0:  # 将没有数据用于训练时，直接退出
        return

    non_final_next_state_tensor = Variable(torch.tensor(non_final_next_state_list, device=device, dtype=torch.float32))

    # 获取所有的reward数据
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # 获取所有的action数据
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # 使用复制的model计算所有的Q(s_t+1, a_t+1)，并且假设在s_t+1为最终的state的时候，Q(s_t+1, a_t+1)=0
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    next_state_action_value[non_final_mask] = duplicated_model(
        non_final_next_state_tensor).max(1)[0]  # 获取maximum state action value，并将为Non-final的state value赋值

    # 使用原始的model计算所有的Q(s_t, a_t)，并且按照选择的action，将对应的action value选出来
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)

    # 给两个model分别设置一个Loss Function，将Model1的输出detach，当做Model2的label，计算Loss和Gradient，
    # 将Model2的输出detach，当做Model1的label，计算Loss和True Gradient
    criterion = nn.MSELoss()
    loss = criterion(current_state_action_value, (reward_tensor + gamma * next_state_action_value.detach()).view(-1, 1))  # 以current state value作为学习对象
    loss.backward()  # 计算gradient

    # 对当前的模型进行优化，注意这里fcnn_model的parameter变了，所以之后只能使用其grad，因为当前的theta已经是theta_t+1
    optimizer.step()

    # 返回当前函数的loss
    return loss.detach().to("cpu").numpy().tolist()


def training_fcnn_model_with_true_gradient(fcnn_model, optimizer, input_data, device, gamma):
    """
    本函数用于针对一个模型，一次训练数据，进行一次训练。本函数使用Bellman Error的True Gradient进行训练

    :param input_data: 是一个List of Tuple，每个Tuple里面包含的内容有（s_t, a_t, s_t+1, r_t），其维度分别为为
        s_t: (160, 160, 3) a_t : (3, 1) s_t+1: (160, 160, 3) r_t: (1, )
    :param fcnn_model: 需要训练的模型，已经放到Device上了。这里device指CPU或者GPU。
    :param optimizer: 当前模型的优化器，需要是全局性的，因为里面可能会包含类似Momentum之类的全局信息
    :param device:
    :param gamma: Discount Factor，期望奖励的折扣因子
    :return:
    """
    # 将前一次Tensor中的grad清零，以便于后续的运算
    optimizer.zero_grad()

    # 将model复制一份,用于之后的计算
    duplicated_model = FCNN(input_size=fcnn_model.input_size, output_size=fcnn_model.output_size)
    duplicated_model.load_state_dict(fcnn_model.state_dict())
    duplicated_model.to(device)

    # 获取所有的s_t数据，处理后放到device上
    current_state_list = []
    for data_tuple in input_data:
        current_state_list.append(data_tuple[0])
    current_state_tensor = Variable(torch.tensor(current_state_list, device=device, dtype=torch.float32))

    # 获取所有的s_t+1数据，并挑选出那些不为final state的状态作为函数的输入
    non_final_next_state_list = []
    next_state_list = []
    for data_tuple in input_data:
        next_state_list.append(data_tuple[2])
        if data_tuple[2] is not None:
            non_final_next_state_list.append(data_tuple[2])

    if len(non_final_next_state_list) == 0:  # 将没有数据用于训练时，直接退出
        return

    non_final_next_state_tensor = Variable(torch.tensor(
        non_final_next_state_list, device=device, dtype=torch.float32))

    # 获取所有的reward数据
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # 获取所有的action数据
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # 使用原始的model计算所有的Q(s_t, a_t)，并且按照选择的action，将对应的action value选出来
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)

    # 使用复制的model计算所有的Q(s_t+1, a_t+1)，并且假设在s_t+1为最终的state的时候，Q(s_t+1, a_t+1)=0
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    next_state_action_value[non_final_mask] = duplicated_model(
        non_final_next_state_tensor).max(1)[0]  # 获取maximum state action value，并将为Non-final的state value赋值

    # 给两个model分别设置一个Loss Function，将Model1的输出detach，当做Model2的label，计算Loss和Gradient，
    # 将Model2的输出detach，当做Model1的label，计算Loss和True Gradient
    # 这里计算 [f(x1) - (r + a * v(x2))]^2的loss和gradient
    criterion1 = nn.MSELoss()
    target1 = reward_tensor + gamma * next_state_action_value.detach()
    loss1 = criterion1(current_state_action_value.squeeze(-1),
                       target1)  # 以current state value作为学习对象
    loss1.backward()  # 计算gradient

    # 这里计算[a * f(x2) - ( -r + v(x1) )]^2的loss和gradient
    criterion2 = nn.MSELoss()
    target2 = - reward_tensor + current_state_action_value.detach().squeeze(-1)
    loss2 = criterion2(gamma * next_state_action_value,
                       target2)  # 以next state value function作为学习对象
    loss2.backward()  # 计算gradient

    # 对原来Model中的每一个Parameter（Tensor），进行Gradient的修改，获得Temporal Difference Error的真正的gradient
    # 表示为 [f(x1) - r - a * v(x2)]grad(f(x1)) + [a * f(x2) - ( -r + v(x1) )]grad(f(x2))
    # = [v(x1) - r - a * v(x2)]*[grad(f(x1)) - grad(f(x2))]
    for param1, param2 in zip(fcnn_model.parameters(), duplicated_model.parameters()):
        param1.grad += param2.grad

    # 对当前的模型进行优化
    optimizer.step()

    # 返回当前函数的loss
    return loss1.detach().to("cpu").numpy().tolist()


def save_model(model, filename, state_representation_size, action_size, save_file_path):
    torch.save({"model": model.state_dict(), "state_representation_size": state_representation_size,
                "action_size": action_size},
               save_file_path + "/" + filename + ".pl")


def load_model(filename, save_file_path):
    if not os.path.exists(save_file_path + "/" + filename + ".pl"):
        print("can't load model")
        return None

    state_dict = torch.load(save_file_path + "/" + filename + ".pl")
    model = FCNN(input_size=state_dict["state_representation_size"], output_size=state_dict["action_size"])
    model.load_state_dict(state_dict["model"])
    return model


class MDPAgent(object):
    def __init__(self, buffer_size, state_representation_size, action_size, optimizer_type,
                 init_learning_rate, gradient_type, lr_discount_factor, lr_discount_epoch, training_num,
                 save_file_path):

        self.replay_buffer = SimpleReplayBuffer(buffer_size=buffer_size)

        self.gradient_type = gradient_type
        self.training_num = training_num
        self.state_representation_size = state_representation_size
        self.action_size = action_size

        # 如果存在已经训练的模型，则读取模型，否则新建模型
        model = load_model("fcnn_" + gradient_type, save_file_path=save_file_path)
        if model is None:
            self.model = FCNN(input_size=state_representation_size, output_size=action_size)
            self.model = self.model.to(DEVICE)
        else:
            print("---------------------------load model-----------------------------")
            self.model = model.to(DEVICE)

        self.learning_rate = init_learning_rate
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Unrecognized optimizer type: " + optimizer_type)

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == "sgd":
            momentum = 0.8
            dampening = 0.1
            # momentum = 0
            # dampening = 0
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum,
                                             dampening=dampening)
        else:
            raise Exception("Unrecognized optimizer type: " + optimizer_type)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=int(lr_discount_epoch/self.training_num), gamma=lr_discount_factor)

    def reset_optimizer(self):
        self.optimizer.zero_grad()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def get_current_policy_and_value(self, state_representation):

        state_representation_tensor = torch.tensor(state_representation, device=DEVICE, dtype=torch.float32)
        current_value, current_policy = self.model(state_representation_tensor).max(1)
        policy = current_policy.cpu().detach().numpy().tolist()
        state_value_function = current_value.cpu().detach().numpy().tolist()

        return policy, state_value_function

    def offline_learning(self, batch_size, gamma, gradient_type):


        sample_data = self.replay_buffer.get_sequential_batch_data(batch_size)

        if gradient_type == "true":
            loss = training_fcnn_model_with_true_gradient(fcnn_model=self.model, device=DEVICE,
                                                       input_data=sample_data, optimizer=self.optimizer,
                                                       gamma=gamma)
        elif gradient_type == "semi":
            loss = training_fcnn_model_with_semi_gradient(fcnn_model=self.model, device=DEVICE,
                                                                  input_data=sample_data, optimizer=self.optimizer,
                                                                  gamma=gamma)
        else:
            raise Exception("can't recognize gradient_type: " + self.gradient_type)

        self.lr_scheduler.step()
        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_func(horizon_length, gamma, optimizer_type, save_file_path,
              init_learning_rate, training_num, gradient_type,
              lr_discount_factor, lr_discount_epoch, mdp_random_seed, nn_random_seed):
    # 控制模型的初始化
    setup_seed(mdp_random_seed)

    # 对环境进行初始化
    state_representation_size = 15
    action_size = 4
    deterministic_env = DeterministicMDP(state_representation_size=state_representation_size, max_step=horizon_length,
                                         save_file_path=save_file_path)

    # 控制NN的初始化
    setup_seed(nn_random_seed)

    # 读取并整理训练数据
    training_state_size = 11
    observed_state_size = 15
    training_data = deterministic_env.get_partial_data_tuple(partial_state_size=training_state_size)
    training_data_dict_list = []
    reward_list = []
    next_state_list = []
    action_list = []
    current_state_list = []
    for data in training_data:
        current_state_list.append(data[0])
        action_list.append(data[1])
        next_state_list.append(data[2])
        reward_list.append(data[3])

        if data[4] is False:
            training_data_dict_list.append({"current_state": data[0], "next_state": data[2],
                                            "current_action": data[1], "reward": data[3]})
        else:
            training_data_dict_list.append({"current_state": data[0], "next_state": None,
                                            "current_action": data[1], "reward": data[3]})

    semi_loss_list = []
    true_loss_list = []

    true_state_value_function_list = []
    semi_state_value_function_list = []

    policy_list = []

    # 初始化超参数
    buffer_size = batch_size = len(training_data)

    # 初始化agent对象
    mdp_agent = MDPAgent(buffer_size=buffer_size, state_representation_size=15,
                         action_size=4, optimizer_type=optimizer_type,
                         init_learning_rate=init_learning_rate, gradient_type=gradient_type,
                         lr_discount_factor=lr_discount_factor, lr_discount_epoch=lr_discount_epoch,
                         training_num=training_num, save_file_path=save_file_path)
    mdp_agent.replay_buffer.insert_data_tuple_list(training_data_dict_list)

    for current_epoch in range(10000):

        # 获取每一轮的policy，l1 loss以及状态值函数
        policy, state_value_function = mdp_agent.get_current_policy_and_value(
            state_representation=deterministic_env.state_representation[:observed_state_size, :]
        )

        policy_list.append(policy)
        true_state_value_function_list.append(state_value_function)

        print("current mean value is " + str(np.mean(state_value_function)))
        print("current policy is " + str(policy))

        current_loss = mdp_agent.offline_learning(batch_size=batch_size, gamma=gamma, gradient_type="true")

        true_loss_list.append(current_loss)

        print("finish one epoch of training ********** " + str(current_loss) + ", current epoch is " + str(current_epoch))

        save_model(model=mdp_agent.model, filename="fcnn_" + gradient_type + "_true",
                   state_representation_size=state_representation_size, action_size=action_size,
                   save_file_path=save_file_path)

    mdp_agent.reset_optimizer()

    for current_epoch in range(15000):

        # 获取每一轮的policy，l1 loss以及状态值函数
        policy, state_value_function = mdp_agent.get_current_policy_and_value(
            state_representation=deterministic_env.state_representation[:observed_state_size, :]
        )

        policy_list.append(policy)
        semi_state_value_function_list.append(state_value_function)

        print("current mean value is " + str(np.mean(state_value_function)))
        print("current policy is " + str(policy))

        current_loss = mdp_agent.offline_learning(batch_size=batch_size, gamma=gamma, gradient_type="semi")
        semi_loss_list.append(current_loss)
        print("finish one epoch of training ********** " + str(current_loss) + ", current epoch is " + str(current_epoch))

        save_model(model=mdp_agent.model, filename="fcnn_" + gradient_type + "_semi",
                   state_representation_size=state_representation_size, action_size=action_size,
                   save_file_path=save_file_path)

    # 可视化loss
    plot_log_loss(true_loss_list=true_loss_list, semi_loss_list=semi_loss_list, file_save_path=save_file_path)
    # 可视化每一个状态的值函数的变化
    plot_value_function(semi_state_value_tuple_list=semi_state_value_function_list,
                        true_state_value_tuple_list=true_state_value_function_list, semi_loss_list=semi_loss_list,
                        true_loss_list=true_loss_list, file_save_path=save_file_path, state_size=observed_state_size)
    # 可视化policy的变化
    # plot_policies(policy_list=policy_list, semi_loss_list=semi_loss_list, true_loss_list=true_loss_list,
    #               state_size=observed_state_size, file_save_path=save_file_path)
    plot_partial_policies(policy_list=policy_list, semi_loss_list=semi_loss_list, true_loss_list=true_loss_list,
                  state_size=observed_state_size, file_save_path=save_file_path, start_idx=16000, end_idx=17000)


    # 将输出的数据存入pickle中
    pickle.dump(
        {
            "true_loss_list": true_loss_list,
            "semi_loss_list": semi_loss_list,
            "true_state_value_function_list": true_state_value_function_list,
            "semi_state_value_function_list": semi_state_value_function_list,
            "policy_list": policy_list
        },
        open(save_file_path + "/" + gradient_type + "_output_result.pl", "wb")
    )

    deterministic_env.close()

    return buffer_size, batch_size


if __name__ == "__main__":

    # import pickle
    #
    # my_file = pickle.load(open("interchange_true_semi_output_result.pl", "rb"))
    #
    # # 可视化loss
    # plot_log_loss(true_loss_list=my_file["true_loss_list"], semi_loss_list=my_file["semi_loss_list"], file_save_path=".")
    # # 可视化每一个状态的值函数的变化
    # plot_value_function(semi_state_value_tuple_list=my_file["semi_state_value_function_list"],
    #                     true_state_value_tuple_list=my_file["true_state_value_function_list"], semi_loss_list=my_file["semi_loss_list"],
    #                     true_loss_list=my_file["true_loss_list"], file_save_path=".", state_size=15)
    # # 可视化policy的变化
    # plot_partial_policies(policy_list=my_file["policy_list"], semi_loss_list=my_file["semi_loss_list"], true_loss_list=my_file["true_loss_list"],
    #                       state_size=15, file_save_path=".", start_idx=16000, end_idx=17000)

    # 获得超参
    TRAINING_NUM = 1  # 每次采样完成后的训练次数

    # 用于指示当前一次探索应该采样多少次
    HORIZON_LENGTH = 20

    OPTIMIZER_TYPE = "sgd"  # 规定优化器类型，现在支持“adam”和“sgd”

    INIT_LEARNING_RATE = 0.3  # 规定学习率

    GAMMA = 0.98  # 奖励折扣因子
    MDP_RANDOM_SEED = 4  # state embedding的随机种子
    NN_RANDOM_SEED = 75  # 神经网络初始化的随机种子

    LR_DISCOUNT_EPOCH = 3000  # 学习率下降步长
    LR_DISCOUNT_FACTOR = 1  # 学习率下降比例

    BUFFER_SIZE = None
    BATCH_SIZE = None

    # 创建保存模型和结果的文件夹
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "_grid_world_true_to_semi"
    SAVE_FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/result/" + folder_name
    os.makedirs(SAVE_FILE_PATH)
    copyfile(CURRENT_PATH + "/q_learning_true_to_semi_grid_world.py", SAVE_FILE_PATH + "/q_learning_true_to_semi_grid_world.py")

    for GRADIENT_TYPE in ["interchange_true_semi"]:
    # for GRADIENT_TYPE in ["interchange_true_semi", "semi", "true", "target_network"]:
        BUFFER_SIZE, BATCH_SIZE = main_func(
            init_learning_rate=INIT_LEARNING_RATE, gamma=GAMMA, optimizer_type=OPTIMIZER_TYPE,
            horizon_length=HORIZON_LENGTH, training_num=TRAINING_NUM,
            gradient_type=GRADIENT_TYPE, lr_discount_epoch=LR_DISCOUNT_EPOCH, lr_discount_factor=LR_DISCOUNT_FACTOR,
            mdp_random_seed=MDP_RANDOM_SEED, nn_random_seed=NN_RANDOM_SEED, save_file_path=SAVE_FILE_PATH)

        # 将传入的超参写入txt中
        HYPER_PARAM_DICT = {
            "horizon length": HORIZON_LENGTH,
            "gamma": GAMMA,
            "optimizer type": OPTIMIZER_TYPE,
            "initial learning rate": INIT_LEARNING_RATE,
            "training number in each epoch": TRAINING_NUM,
            "learning rate discount epoch": LR_DISCOUNT_EPOCH,
            "learning rate discount factor": LR_DISCOUNT_FACTOR,
            "mdp random seed": MDP_RANDOM_SEED,
            "nn random seed": NN_RANDOM_SEED,
            "buffer size": BUFFER_SIZE,
            "batch size": BATCH_SIZE,
        }
        with open(SAVE_FILE_PATH + "/hyper_param.json", "w") as f:
            f.write(str(HYPER_PARAM_DICT))
