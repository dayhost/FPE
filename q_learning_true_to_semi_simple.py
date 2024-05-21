import os
import datetime
import torch
import random
import numpy as np
import pickle
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
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


def plot_policies(policy_list, true_loss_list, semi_loss_list, file_save_path):
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    total_policy_pd = pd.DataFrame()
    for i in range(len(policy_list)):
        policy_dict = {"iter" + str(i): {}}
        for j in range(len(policy_list[i])):
            policy_dict["iter" + str(i)][r"$s_ " + str(j+1) + "$"] = policy_list[i][j]
        total_policy_pd = pd.concat([total_policy_pd, pd.DataFrame().from_dict(policy_dict)], axis=1)

    myColors = ((231/255, 86/255, 59/255, 1), (106/255, 7/255, 10/255, 1))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    ax = sns.heatmap(total_policy_pd, cmap=cmap)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels([r'$a_1$', r'$a_2$'])

    x_label = list(np.arange(0, len(policy_list)))

    ax.set_xticks(range(len(x_label)))
    ax.set_xticklabels(x_label)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='y', labelrotation=0)

    plt.plot(x_label, [1] * len(x_label), color="white")
    plt.plot(x_label, [2] * len(x_label), color="white")

    plt.axvline(mid_idx, color="white", linestyle="--")
    plt.axvline(max_after_mid_idx, color="white", linestyle="--")
    plt.text(x=mid_idx - 2000, y=1.5, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")
    plt.text(x=max_after_mid_idx - 2000, y=1.5, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'}, color="white")

    plt.xlabel("training step")
    plt.ylabel("state")
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_dqn_policy_change.png")
    plt.close()


def plot_loss(true_loss_list, semi_loss_list, file_save_path):
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    print("max_after_mid_idx: " + str(max_after_mid_idx))

    plt.plot(np.arange(0, len(true_loss_list), 1), true_loss_list, color="red")
    plt.plot(np.arange(len(true_loss_list), len(true_loss_list) + len(semi_loss_list), 1), semi_loss_list, color="blue")

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx - 2000, y=0.011, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx - 2000, y=0.011, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    # plt.scatter(max_after_mid_idx, max(semi_loss_list), marker="X", s=200, color="orange")

    plt.xlabel("training step")
    plt.ylabel("loss")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    plt.tight_layout(pad=0)
    plt.savefig(file_save_path + "/true_semi_dqn_loss.png")
    plt.close()


def plot_value_function(true_state_value_tuple_list, semi_state_value_tuple_list, true_loss_list, semi_loss_list, file_save_path):
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][0])
    l1, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_1$",
                   color="green")

    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][1])
    l2, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_2$",
                   color="purple")

    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][2])
    l3, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_3$",
                   color="brown")

    # 开始画State Value的Semi State的线
    data_dict_list = []
    for i in range(len(semi_state_value_tuple_list)):
        data_dict_list.append(semi_state_value_tuple_list[i][0])
    l4, = plt.plot(list(
        np.arange(len(true_state_value_tuple_list), len(true_state_value_tuple_list) + len(semi_state_value_tuple_list),
                  1)), data_dict_list, label=r"$s_1$", color="green",
        linestyle="--")

    data_dict_list = []
    for i in range(len(semi_state_value_tuple_list)):
        data_dict_list.append(semi_state_value_tuple_list[i][1])
    l5, = plt.plot(list(
        np.arange(len(true_state_value_tuple_list), len(true_state_value_tuple_list) + len(semi_state_value_tuple_list),
                  1)), data_dict_list, label=r"$s_2$", color="purple",
        linestyle="--")

    data_dict_list = []
    for i in range(len(semi_state_value_tuple_list)):
        data_dict_list.append(semi_state_value_tuple_list[i][2])
    l6, = plt.plot(list(
        np.arange(len(true_state_value_tuple_list), len(true_state_value_tuple_list) + len(semi_state_value_tuple_list),
                  1)), data_dict_list, label=r"$s_3$", color="brown",
        linestyle="--")

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx - 2000, y=-0.33, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx - 2000, y=-0.33, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    plt.legend(handles=[(l1, l4), (l2, l5), (l3, l6)], labels=[r'$s_1$', r'$s_2$', r'$s_3$'],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.xlabel("training step")
    plt.ylabel("state value")
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.tight_layout(pad=0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4000))
    plt.savefig(file_save_path + "/true_semi_value_function_for_each_state_dqn.png")
    plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



class FCNN(nn.Module):
    def __init__(self, input_size, output_size):
        width = 100
        super(FCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, output_size)

        self.fc1.weight.data.fill_(160/width)
        self.fc1.bias.data.fill_(-0.1/width)
        self.fc2.weight.data.fill_(1/width)
        self.fc2.bias.data.fill_(-0.1/width)

    def forward(self, data, action_list=None):
        x = functional.relu(self.fc1(data))
        x = self.fc2(x)

        if action_list is not None:
            if len(x.shape) == 1:
                return x.gather(0, action_list)
            else:
                return x.gather(1, action_list)
        else:
            return x


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


def training_model_with_semi_gradient(model, optimizer, input_data, device, gamma):
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
    # 获取maximum state action value，并将为Non-final的state value赋值
    next_state_action_value[non_final_mask] = model(non_final_next_state_tensor).max(1)[0]

    # 使用原始的model计算所有的Q(s_t, a_t)，并且按照选择的action，将对应的action value选出来
    current_state_action_value_raw = model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)

    criterion = nn.MSELoss()
    loss = criterion(current_state_action_value, (reward_tensor + gamma * next_state_action_value.detach()).view(-1, 1))
    loss.backward()  # 计算gradient

    # 对当前的模型进行优化，注意这里fcnn_model的parameter变了，所以之后只能使用其grad，因为当前的theta已经是theta_t+1
    optimizer.step()

    loss_num = loss.data.to("cpu").numpy().tolist()

    # 返回当前函数的loss
    return loss_num


def training_model_with_true_gradient(model, optimizer, input_data, device, gamma):
    """
    本函数用于针对一个模型，一次训练数据，进行一次训练。本函数使用Bellman Error的True Gradient进行训练

    :param input_data: 是一个List of Tuple，每个Tuple里面包含的内容有（s_t, a_t, s_t+1, r_t），其维度分别为为
        s_t: (160, 160, 3) a_t : (3, 1) s_t+1: (160, 160, 3) r_t: (1, )
    :param model: 需要训练的模型，已经放到Device上了。这里device指CPU或者GPU。
    :param optimizer: 当前模型的优化器，需要是全局性的，因为里面可能会包含类似Momentum之类的全局信息
    :param device:
    :param gamma: Discount Factor，期望奖励的折扣因子
    :return:
    """
    # 将前一次Tensor中的grad清零，以便于后续的运算
    optimizer.zero_grad()

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
    current_state_action_value_raw = model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)

    # 使用复制的model计算所有的Q(s_t+1, a_t+1)，并且假设在s_t+1为最终的state的时候，Q(s_t+1, a_t+1)=0
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    next_state_action_value[non_final_mask] = model(
        non_final_next_state_tensor).max(1)[0]  # 获取maximum state action value，并将为Non-final的state value赋值

    # 给两个model分别设置一个Loss Function，将Model1的输出detach，当做Model2的label，计算Loss和Gradient，
    # 将Model2的输出detach，当做Model1的label，计算Loss和True Gradient
    # 这里计算 [f(x1) - (r + a * v(x2))]^2的loss和gradient
    criterion = nn.MSELoss()
    target1 = reward_tensor + gamma * next_state_action_value.detach()
    loss1 = criterion(current_state_action_value.squeeze(-1), target1)  # 以current state value作为学习对象

    # 这里计算[a * f(x2) - ( -r + v(x1) )]^2的loss和gradient
    target2 = - reward_tensor + current_state_action_value.detach().squeeze(-1)
    loss2 = criterion(gamma * next_state_action_value, target2)  # 以next state value function作为学习对象

    total_loss = loss1 + loss2
    total_loss.backward()

    # 对当前的模型进行优化，注意这里fcnn_model的parameter变了，所以之后只能使用其grad，因为当前的theta已经是theta_t+1
    optimizer.step()

    loss_num = loss1.detach().to("cpu").numpy().tolist()

    # 返回当前函数的loss
    return loss_num


def save_model(model, optimizer, lr_scheduler, filename, file_path):
    torch.save({"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler},
               file_path + "/" + filename + ".pl")


def load_model(filename, file_path):
    if not os.path.exists(file_path + "/" + filename + ".pl"):
        print("can't load model")
        return None, None, None

    state_dict = torch.load(file_path + "/" + filename + ".pl")
    return state_dict["model"], state_dict["optimizer"], state_dict["lr_scheduler"]


def main_func(optimizer_type, init_learning_rate, random_seed, save_file_path):
    # 控制模型的初始化
    setup_seed(random_seed)

    # 控制是否适用cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    gamma = 0.9
    phi_s1 = 0.1
    phi_s2 = phi_s1 / gamma - 0.05
    phi_s3 = phi_s1 / gamma + 0.05
    r = -0.1

    training_data_dict_list = []
    # 从给定文件夹内读取已有的MDP环境
    training_data_dict_list.append({"current_state": [phi_s1], "next_state": [phi_s2],
                                    "current_action": 0, "reward": r})
    training_data_dict_list.append({"current_state": [phi_s1], "next_state": [phi_s3],
                                    "current_action": 1, "reward": r})

    buffer_size = batch_size = 2

    replay_buffer = SimpleReplayBuffer(buffer_size=buffer_size)
    replay_buffer.insert_data_tuple_list(training_data_dict_list)

    model = FCNN(input_size=1, output_size=2).to(device)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_learning_rate)
    else:
        raise Exception("Unrecognized optimizer type: " + optimizer_type)

    # 观测变量数组
    true_state_value_tuple_list = []
    semi_state_value_tuple_list = []
    true_loss_list = []
    semi_loss_list = []

    total_policy_list = []

    for current_epoch in range(10000):

        data_list = replay_buffer.get_sequential_batch_data(batch_size=batch_size)
        loss = training_model_with_true_gradient(model=model, optimizer=optimizer, input_data=data_list,
                                                 gamma=gamma, device=device)

        print("step: " + str(current_epoch) + "     loss: " + str(loss))

        save_model(model=model, optimizer=optimizer, lr_scheduler=None,
                   filename="true_to_semi_model", file_path=save_file_path)

        current_state_tensor = torch.tensor([[phi_s1], [phi_s2], [phi_s3]], dtype=torch.float32, device=device)
        state_action_value_tensor = model(current_state_tensor)
        state_value_tensor, policy_tensor = state_action_value_tensor.max(1)
        state_value_list = state_value_tensor.cpu().detach().numpy().tolist()
        policy_list = policy_tensor.cpu().detach().numpy().tolist()

        # 观测变量入数组
        true_loss_list.append(loss)
        true_state_value_tuple_list.append(state_value_list)
        total_policy_list.append(policy_list)

    for current_epoch in range(10000):

        data_list = replay_buffer.get_sequential_batch_data(batch_size=batch_size)
        loss = training_model_with_semi_gradient(model=model, optimizer=optimizer, input_data=data_list,
                                                 gamma=gamma, device=device)

        print("step: " + str(current_epoch) + "     loss: " + str(loss))

        save_model(model=model, optimizer=optimizer, lr_scheduler=None,
                   filename="true_to_semi_model", file_path=save_file_path)

        current_state_tensor = torch.tensor([[phi_s1], [phi_s2], [phi_s3]], dtype=torch.float32, device=device)
        state_action_value_tensor = model(current_state_tensor)
        state_value_tensor, policy_tensor = state_action_value_tensor.max(1)
        state_value_list = state_value_tensor.cpu().detach().numpy().tolist()
        policy_list = policy_tensor.cpu().detach().numpy().tolist()

        # 观测变量入数组
        semi_loss_list.append(loss)
        semi_state_value_tuple_list.append(state_value_list)
        total_policy_list.append(policy_list)

    # 开始画图
    plot_loss(semi_loss_list=semi_loss_list, true_loss_list=true_loss_list, file_save_path=save_file_path)
    plot_value_function(true_state_value_tuple_list=true_state_value_tuple_list,
                        semi_state_value_tuple_list=semi_state_value_tuple_list,
                        true_loss_list=true_loss_list,
                        semi_loss_list=semi_loss_list,
                        file_save_path=save_file_path)
    plot_policies(policy_list=total_policy_list, true_loss_list=true_loss_list,
                  semi_loss_list=semi_loss_list, file_save_path=save_file_path)

    # 将输出的数据存入pickle中
    pickle.dump(
        {
            "true_loss_list": true_loss_list,
            "semi_loss_list": semi_loss_list,
            "true_state_value_function_list": true_state_value_tuple_list,
            "semi_state_value_function_list": semi_state_value_tuple_list,
            "policy_list": total_policy_list
        },
        open(save_file_path + "/true_2_semi_output_result.pl", "wb")
    )

    return buffer_size, batch_size


if __name__ == "__main__":

        RANDOM_SEED = 2
        # 用于指示当前一次探索应该采样多少次
        HORIZON_LENGTH = 20

        OPTIMIZER_TYPE = "sgd"  # 规定优化器类型，现在支持“adam”和“sgd”

        INIT_LEARNING_RATE = 2e-3  # 规定学习率

        BUFFER_SIZE = None
        BATCH_SIZE = None

        # 创建保存模型和结果的文件夹
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "_true_to_semi"
        SAVE_FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/result/" + folder_name
        os.makedirs(SAVE_FILE_PATH)
        copyfile(CURRENT_PATH + "/q_learning_true_to_semi_simple.py", SAVE_FILE_PATH + "/q_learning_true_to_semi_simple.py")

        BUFFER_SIZE, BATCH_SIZE = main_func(init_learning_rate=INIT_LEARNING_RATE,
                                            random_seed=RANDOM_SEED, save_file_path=SAVE_FILE_PATH,
                                            optimizer_type=OPTIMIZER_TYPE)

        # 将传入的超参写入txt中
        HYPER_PARAM_DICT = {
            "optimizer type": OPTIMIZER_TYPE,
            "initial learning rate": INIT_LEARNING_RATE,
            "random seed": RANDOM_SEED,
            "buffer size": BUFFER_SIZE,
            "batch size": BATCH_SIZE,
        }
        with open(SAVE_FILE_PATH + "/hyper_param.json", "w") as f:
            f.write(str(HYPER_PARAM_DICT))
