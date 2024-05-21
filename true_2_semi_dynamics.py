import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from fplanck import fokker_planck, boundary, gaussian_pdf
import matplotlib
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (6, 5)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerTuple

# 定义超参数
gamma = 0.9
r_s1_a1 = -0.1
r_s1_a2 = -0.1
phi_s1 = 0.1
phi_s2 = phi_s1 / gamma - 0.05
phi_s3 = 0.1
phi_s4 = phi_s3 / gamma + 0.05


def calculate_gradient_from_stationary_distribution(theta_a1_list, theta_a2_list, stationary_distribution, sample_rate):
    gradient_x1_matrix = np.zeros((len(theta_a1_list), len(theta_a1_list[0])))
    gradient_x2_matrix = np.zeros((len(theta_a1_list), len(theta_a1_list[0])))

    theta_a1_interval = theta_a1_list[1][0] - theta_a1_list[0][0]
    theta_a2_interval = theta_a2_list[0][1] - theta_a2_list[0][0]

    for i in range(len(theta_a1_list)):
        for j in range(len(theta_a1_list[0])):

            if i < len(theta_a1_list) - 1 and i > 0 and j < len(theta_a1_list[0]) - 1 and j > 0:
                gradient_x1_matrix[i][j] = (1/stationary_distribution[i][j]) * \
                                           ((stationary_distribution[i + 1][j] - stationary_distribution[i - 1][j]) / (2 * theta_a1_interval))
                gradient_x2_matrix[i][j] = (1/stationary_distribution[i][j]) * \
                                           ((stationary_distribution[i][j + 1] - stationary_distribution[i][j - 1]) / (2 * theta_a2_interval))
            else:
                gradient_x1_matrix[i][j] = 0

    return np.array([gradient_x1_matrix[::sample_rate, ::sample_rate], gradient_x2_matrix[::sample_rate, ::sample_rate]])


# 根据等效Stationary Distribution计算出Flux
def calculate_flux_of_coordinate(theta_a1_list, theta_a2_list, force_x1, force_x2, wanted_diffusion_constant, stationary_distribution, sample_rate=1):
    """
    Flux在每一个维度上的定义为
    F_1(x_1, x_2) = Force_1(x_1, x_2) * P(x_1, x_2) + Diffusion * (P(x_1 + delta x, x_2) - P(x_1, x_2))/delta x
    F_2(x_1, x_2) = Force_2(x_1, x_2) * P(x_1, x_2) + Diffusion * (P(x_1, x_2 + delta x) - P(x_1, x_2))/delta x
    """

    flux_x1_matrix = np.zeros((len(theta_a1_list), len(theta_a1_list[0])))
    flux_x2_matrix = np.zeros((len(theta_a1_list), len(theta_a1_list[0])))

    theta_a1_interval = theta_a1_list[1][0] - theta_a1_list[0][0]
    theta_a2_interval = theta_a2_list[0][1] - theta_a2_list[0][0]

    for i in range(len(theta_a1_list)):
        for j in range(len(theta_a1_list[0])):

            if i < len(theta_a1_list) - 1 and i > 0 and j < len(theta_a1_list[0]) - 1 and j > 0:
                flux_x1_matrix[i][j] = force_x1[i][j] * stationary_distribution[i][j] - wanted_diffusion_constant * \
                                       (stationary_distribution[i + 1][j] - stationary_distribution[i - 1][j]) / (2 * theta_a1_interval)
                flux_x2_matrix[i][j] = force_x2[i][j] * stationary_distribution[i][j] - wanted_diffusion_constant * \
                                       (stationary_distribution[i][j + 1] - stationary_distribution[i][j - 1]) / (2 * theta_a2_interval)
            else:
                flux_x1_matrix[i][j] = 0

    return np.array([flux_x1_matrix[::sample_rate, ::sample_rate], flux_x2_matrix[::sample_rate, ::sample_rate]])


def divide_force_into_flux_and_gradient(stationary_distribution, gradient, flux, wanted_diffusion_constant, sample_rate):

    divide_gradient = np.multiply(wanted_diffusion_constant, gradient)
    divide_gradient = np.array([divide_gradient[0][::sample_rate, ::sample_rate],
                                divide_gradient[1][::sample_rate, ::sample_rate]])

    divide_flux = [np.divide(flux[0], stationary_distribution), np.divide(flux[1], stationary_distribution)]
    divide_flux = np.array([divide_flux[0][::sample_rate, ::sample_rate],
                            divide_flux[1][::sample_rate, ::sample_rate]])

    return divide_gradient, divide_flux


# 根据Loss Landscape计算Critical Point的位置
def calculate_critical_point(theta_a1_list, theta_a2_list, gradient_x1, gradient_x2):

    min_i = 0
    min_j = 0
    min_norm = 10
    for i in range(len(theta_a1_list)):
        for j in range(len(theta_a1_list[0])):

            gradient_norm = np.linalg.norm([gradient_x1[i][j], gradient_x2[i][j]])

            if gradient_norm > 0 and gradient_norm < min_norm:
            # if gradient_norm < min_norm:
                min_norm = gradient_norm
                min_i = i
                min_j = j


    return [theta_a1_list[min_i][min_j], theta_a2_list[min_i][min_j]]


# 产生一个把Stationary Distribution算出来为0的部分标记出来的mask
def generate_mask_by_stationary_distribution(stationary_distribution):
    mask_matrix = np.zeros([len(stationary_distribution), len(stationary_distribution[0])])
    for i in range(len(stationary_distribution)):
        for j in range(len(stationary_distribution[0])):

            if stationary_distribution[i][j] == 0:
                mask_matrix[i][j] = 0
            else:
                mask_matrix[i][j] = 1

    return mask_matrix

def generate_mask_by_vector_length(gradient_x, gradient_y, flux_x, flux_y, vector_scale):
    mask_matrix = np.zeros([len(gradient_x), len(gradient_x[0])])
    for i in range(len(gradient_x)):
        for j in range(len(gradient_x[0])):

            gradient_norm = np.linalg.norm([gradient_x[i][j], gradient_y[i][j]])
            flux_norm = np.linalg.norm([flux_x[i][j], flux_y[i][j]])

            if (gradient_norm >= 100 * vector_scale or gradient_norm == 0) or \
                    (flux_norm >= 100 * vector_scale or flux_norm == 0):
                mask_matrix[i][j] = 0
            else:
                mask_matrix[i][j] = 1

    return mask_matrix


# 定义Force对应的函数，输入为两个位置的值，输出为numpy array
def true_gradient_force(theta_a1_list, theta_a2_list, sample_rate=1):
    """
    当前给定的例子的Stationary Distribution = alpha * exp(-U/D)
    D表示Diffusion Constant，其中的Potential表示为
        U = 1/2 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a1_list)^2 +
            1/2 * (phi_s1 * theta_a2_list - r_s1_a2 - gamma * phi_s3 * theta_a2_list)^2  (theta_a1 >= theta_a2)
          = 1/2 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a2_list)^2 +
            1/2 * (phi_s1 * theta_a2_list - r_s1_a2 - gamma * phi_s3 * theta_a1_list)^2  (theta_a1 < theta_a2)
    """
    true_gradient_vec_a1_a1 = [
        - ((phi_s1 - gamma * phi_s2)*(phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a1_list) -
           gamma * phi_s4 * (phi_s3 * theta_a2_list - r_s1_a2 - gamma * phi_s4 * theta_a1_list)),
        - (phi_s3 * (phi_s3 * theta_a2_list - r_s1_a2 - gamma * phi_s4 * theta_a1_list))
    ]

    true_gradient_vec_a2_a2 = [
        - (phi_s1 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a2_list)),
        - (-gamma * phi_s2 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a2_list) +
           (phi_s3 - gamma * phi_s4) * (phi_s3 * theta_a2_list - r_s1_a2 - gamma * phi_s4 * theta_a2_list))
    ]

    true_gradient_force = np.zeros((2, len(theta_a1_list), len(theta_a1_list[0])))
    for i in range(len(theta_a1_list)):
        for j in range(len(theta_a1_list[0])):

            theta_a1 = theta_a1_list[i][j]
            theta_a2 = theta_a2_list[i][j]

            if theta_a1 >= theta_a2:
                true_gradient_force[0][i][j] = true_gradient_vec_a1_a1[0][i][j]
                true_gradient_force[1][i][j] = true_gradient_vec_a1_a1[1][i][j]
            else:
                true_gradient_force[0][i][j] = true_gradient_vec_a2_a2[0][i][j]
                true_gradient_force[1][i][j] = true_gradient_vec_a2_a2[1][i][j]

    # return true_gradient_force
    return np.array([true_gradient_force[0][::sample_rate, ::sample_rate], true_gradient_force[1][::sample_rate, ::sample_rate]])


# 定义Force对应的函数，输入为两个位置的值，输出为numpy array
def semi_gradient_force(theta_a1_list, theta_a2_list, sample_rate=1):
    """
    U = 1/2 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a1_list)^2 +
        1/2 * (phi_s1 * theta_a2_list - r_s1_a2 - gamma * phi_s3 * theta_a1_list)^2  (theta_a1 >= theta_a2)
      = 1/2 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a2_list)^2 +
        1/2 * (phi_s1 * theta_a2_list - r_s1_a2 - gamma * phi_s3 * theta_a2_list)^2  (theta_a1 < theta_a2)
    """

    semi_gradient_vec_a1_a1 = [
        -phi_s1 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a1_list),
        -phi_s3 * (phi_s3 * theta_a2_list - r_s1_a2 - gamma * phi_s4 * theta_a1_list)
    ]

    semi_gradient_vec_a2_a2 = [
        -phi_s1 * (phi_s1 * theta_a1_list - r_s1_a1 - gamma * phi_s2 * theta_a2_list),
        -phi_s3 * (phi_s3 * theta_a2_list - r_s1_a2 - gamma * phi_s4 * theta_a2_list)
    ]

    semi_gradient_force = np.zeros((2, len(theta_a1_list), len(theta_a1_list[0])))
    for i in range(len(theta_a1_list)):
        for j in range(len(theta_a1_list[0])):
            theta_a1 = theta_a1_list[i][j]
            theta_a2 = theta_a2_list[i][j]

            if theta_a1 >= theta_a2:
                semi_gradient_force[0][i][j] = semi_gradient_vec_a1_a1[0][i][j]
                semi_gradient_force[1][i][j] = semi_gradient_vec_a1_a1[1][i][j]
            else:
                semi_gradient_force[0][i][j] = semi_gradient_vec_a2_a2[0][i][j]
                semi_gradient_force[1][i][j] = semi_gradient_vec_a2_a2[1][i][j]

    return np.array([semi_gradient_force[0][::sample_rate, ::sample_rate], semi_gradient_force[1][::sample_rate, ::sample_rate]])


def generate_semi_gradient_trajectory(start_point, lr, step_n):

    def generate_semi_gradient(x, y):
        if x >= y:
            semi_gradient = [
                -phi_s1 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * x),
                -phi_s3 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x)
            ]
        else:
            semi_gradient = [
                -phi_s1 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y),
                -phi_s3 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * y)
            ]

        return semi_gradient

    x_list = []
    y_list = []

    for _ in range(step_n):
        semi_gradient = generate_semi_gradient(start_point[0], start_point[1])
        start_point[0] += lr * semi_gradient[0]
        start_point[1] += lr * semi_gradient[1]

        x_list.append(copy.deepcopy(start_point[0]))
        y_list.append(copy.deepcopy(start_point[1]))

    return x_list, y_list


def generate_true_gradient_trajectory(start_point, lr, step_n):

    def generate_true_gradient(x, y):

        if x >= y:
            true_gradient = [
            - ((phi_s1 - gamma * phi_s2) * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * x) -
               gamma * phi_s4 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x)),
            - (phi_s3 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x))
        ]
        else:
            true_gradient = [
            - (phi_s1 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y)),
            - (-gamma * phi_s2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y) +
               (phi_s3 - gamma * phi_s4) * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * y))
        ]

        return true_gradient

    x_list = []
    y_list = []

    for _ in range(step_n):
        true_gradient = generate_true_gradient(start_point[0], start_point[1])
        start_point[0] += lr * true_gradient[0]
        start_point[1] += lr * true_gradient[1]

        x_list.append(copy.deepcopy(start_point[0]))
        y_list.append(copy.deepcopy(start_point[1]))

    return x_list, y_list


def main_func():
    nm = 1e-2  # 定义单位长度

    solution_theta1_a1 = r_s1_a1 / phi_s1 + gamma * (phi_s2 / phi_s1) * r_s1_a2 / (phi_s3 - gamma * phi_s4)
    solution_theta1_a2 = r_s1_a2 / (phi_s3 - gamma * phi_s4)

    solution_theta2_a1 = r_s1_a1 / (phi_s1 - gamma * phi_s2)
    solution_theta2_a2 = r_s1_a2 / phi_s3 + gamma * (phi_s4 / phi_s3) * r_s1_a1 / (phi_s1 - gamma * phi_s2)

    if solution_theta1_a1 > solution_theta1_a2 and solution_theta2_a1 > solution_theta2_a2:
        center = [solution_theta1_a1, solution_theta1_a2]
    elif solution_theta1_a1 < solution_theta1_a2 and solution_theta2_a1 < solution_theta2_a2:
        center = [solution_theta2_a1, solution_theta2_a2]
    else:
        center = [(solution_theta1_a1 + solution_theta2_a1) / 2, (solution_theta1_a2 + solution_theta2_a2) / 2]

    # center = [1 / 8, 5 / 4]
    # center = [0, 0]
    extent = [1100 * nm, 1100 * nm]
    resolution = 11 * nm

    init_pdf_center = (-100 * nm + center[0], -100 * nm + center[1])
    init_pdf_width = 30 * nm
    propagation_time = 100000

    step_level = 1

    wanted_mobility_constant = 1  # 期待的Mobility Constant Coefficient
    # wanted_diffusion_constant_list = [2**-2, 2**-3, 2**-4, 2**-5]
    wanted_diffusion_constant = 2 ** -8

    stationary_dist_minimal_value = 1e-40

    boundary_condition = boundary.reflecting

    drag = 1 / wanted_mobility_constant  # drag = 1/mobility, 我们希望mobility为1
    temperature = wanted_diffusion_constant * drag / constants.k  # 定义 Diffusion = kT/drag，也就是 T = Diffusion * drag / k


    # 开始计算True Gradient Distribution
    true_gradient_fpe = fokker_planck(temperature=temperature, drag=drag, extent=extent, center=center,
                resolution=resolution, boundary=boundary_condition, force=true_gradient_force)

    # 进行单个时刻的运算
    true_gradient_init_pdf = gaussian_pdf(center=init_pdf_center, width=init_pdf_width)
    # 计算True Gradient对应的Stationary Distribution
    true_stationary_dist = true_gradient_fpe.propagate(true_gradient_init_pdf, time=propagation_time)
    true_stationary_dist_volume = np.trapz(
        np.trapz(true_stationary_dist, true_gradient_fpe.grid[0][:, 0], axis=0),
        true_gradient_fpe.grid[1][0, :], axis=0)
    true_stationary_dist = true_stationary_dist / true_stationary_dist_volume
    true_stationary_dist[true_stationary_dist == 0] = stationary_dist_minimal_value

    # 开始计算Semi Gradient Distribution
    semi_gradient_fpe = fokker_planck(temperature=temperature, drag=drag, extent=extent, center=center,
                                      resolution=resolution, boundary=boundary.reflecting, force=semi_gradient_force)
    # 进行单个时刻的运算
    semi_gradient_init_pdf = gaussian_pdf(center=init_pdf_center, width=init_pdf_width)
    # 计算Semi Gradient对应的Stationary Distribution
    semi_stationary_dist = semi_gradient_fpe.propagate(semi_gradient_init_pdf, time=propagation_time)
    semi_stationary_dist_volume = np.trapz(
        np.trapz(semi_stationary_dist, semi_gradient_fpe.grid[0][:, 0], axis=0),
        semi_gradient_fpe.grid[1][0, :], axis=0)
    semi_stationary_dist = semi_stationary_dist / semi_stationary_dist_volume
    semi_stationary_dist[semi_stationary_dist == 0] = stationary_dist_minimal_value
    print("finish calculate distribution")

    # 开始进行先true再Semi Gradient Dynamics的轨迹可视化
    # 定义一个先True Gradient再Semi Gradient的情景
    true_sample_rate = 500
    semi_sample_rate = 300
    true_learning_rate = 0.1
    semi_learning_rate = 0.1
    true_x, true_y = generate_true_gradient_trajectory(start_point=[-2, 1],
                                                       lr=true_learning_rate, step_n=25000)
    semi_x, semi_y = generate_semi_gradient_trajectory(start_point=copy.deepcopy([true_x[-1], true_y[-1]]),
                                                       lr=semi_learning_rate, step_n=25000)

    # 计算loss和state value
    semi_loss_list = []
    semi_state_value_tuple_list = []
    for x, y in zip(semi_x, semi_y):
        if x >= y:
            loss = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * x) ** 2 + \
                   1 / 2 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x) ** 2
            s1_state_value = phi_s1 * x
            s2_state_value = phi_s2 * x
            s3_state_value = phi_s4 * x
        else:
            loss = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y) ** 2 + \
                   1 / 2 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * y) ** 2
            s1_state_value = phi_s1 * x
            s2_state_value = phi_s2 * x
            s3_state_value = phi_s4 * x

        semi_loss_list.append(loss)
        semi_state_value_tuple_list.append((s1_state_value, s2_state_value, s3_state_value))

    true_loss_list = []
    true_state_value_tuple_list = []
    for x, y in zip(true_x, true_y):
        if x >= y:
            loss = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * x) ** 2 + \
                   1 / 2 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x) ** 2
            s1_state_value = phi_s1 * x
            s2_state_value = phi_s2 * x
            s3_state_value = phi_s4 * x
        else:
            loss = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y) ** 2 + \
                   1 / 2 * (phi_s3 * y - r_s1_a2 - gamma * phi_s4 * y) ** 2
            s1_state_value = phi_s1 * x
            s2_state_value = phi_s2 * x
            s3_state_value = phi_s4 * x

        true_loss_list.append(loss)
        true_state_value_tuple_list.append((s1_state_value, s2_state_value, s3_state_value))

    # 找到在True切换到Semi后最大值的位置
    mid_idx = len(true_loss_list)
    max_after_mid_idx = mid_idx + semi_loss_list.index(max(semi_loss_list))

    # 开始画轨迹
    fig = plt.figure()
    theta_a1_list, theta_a2_list = semi_gradient_fpe.grid[0], semi_gradient_fpe.grid[1]
    ax = fig.add_subplot(1, 1, 1)

    # 画出policy变化的虚线
    policy_line_theta_1 = []
    policy_line_theta_2 = []
    for theta_1 in theta_a1_list[:, 0]:
        if theta_1 < min(theta_a2_list[0, :]) or theta_1 > max(theta_a2_list[0, :]):
            continue
        policy_line_theta_1.append(theta_1)
        policy_line_theta_2.append(theta_1)
    ax.plot(policy_line_theta_1, policy_line_theta_2, linestyle="dashed", color="k", label="policy boundary", zorder=3)

    # 可视化轨迹
    ax.plot(true_x, true_y, zorder=3, color="red")
    ax.scatter(true_x[::true_sample_rate], true_y[::true_sample_rate], s=10, zorder=4, label="residual trajectory", color="red")
    ax.plot(semi_x, semi_y, zorder=3, color="blue")
    ax.scatter(semi_x[::semi_sample_rate], semi_y[::semi_sample_rate], s=10, zorder=3, label="semi trajectory", color="blue")

    ax.scatter(semi_x[max_after_mid_idx - mid_idx], semi_y[max_after_mid_idx - mid_idx], s=200, zorder=3, color="orange", marker="X")

    ax.scatter([solution_theta1_a1], [solution_theta1_a2], s=200, marker="*", zorder=2, color="C0")
    ax.scatter([solution_theta2_a1], [solution_theta2_a2], s=200, marker="*", zorder=2, color="C1")
    min_level = np.min(-np.log(semi_stationary_dist))
    max_level = np.max(-np.log(semi_stationary_dist))
    contour = ax.contour(theta_a1_list, theta_a2_list, -np.log(true_stationary_dist), colors="gray",
                         levels=np.arange(min_level, max_level + step_level, step_level), zorder=1)
    # ax.clabel(contour, colors="k")
    ax.set(xlabel=r'$\theta(a_1)$', ylabel=r'$\theta(a_2)$')
    ax.legend()
    plt.tight_layout(pad=0)
    plt.subplots_adjust()
    plt.savefig("true_semi_trajectory_true_landscape.png")
    plt.close()

    # 开始画State Value的True State的线
    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][0])
    l1, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_1$", color="green")

    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][1])
    l2, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_2$", color="purple")

    data_dict_list = []
    for i in range(len(true_state_value_tuple_list)):
        data_dict_list.append(true_state_value_tuple_list[i][2])
    l3, = plt.plot(list(np.arange(0, len(true_state_value_tuple_list), 1)), data_dict_list, label=r"$s_3$", color="brown")

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
    plt.text(x=mid_idx-4000, y=-0.33, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx-4000, y=-0.33, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    plt.legend(handles=[(l1, l4), (l2, l5), (l3, l6)], labels=[r"$s_1$", r"$s_2$", r"$s_3$"], handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.xlabel("training step")
    plt.ylabel("state value")
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.tight_layout(pad=0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
    plt.savefig("true_semi_value_function_for_each_state.png")
    plt.close()

    plt.plot(np.arange(0, len(true_loss_list), 1), true_loss_list, color="red")
    plt.plot(np.arange(len(true_loss_list),len(true_loss_list) + len(semi_loss_list), 1), semi_loss_list,  color="blue")

    plt.axvline(mid_idx, color="black", linestyle="--")
    plt.axvline(max_after_mid_idx, color="black", linestyle="--")
    plt.text(x=mid_idx-4000, y=0.011, s=r"$t_1$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})
    plt.text(x=max_after_mid_idx-4000, y=0.011, s=r"$t_2$",
             fontdict={'fontweight': 'bold', 'fontsize': 18, 'fontname': 'Times New Roman'})

    plt.scatter(max_after_mid_idx, max(semi_loss_list), marker="X", s=200, color="orange")

    plt.xlabel("training step")
    plt.ylabel("loss")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
    plt.tight_layout(pad=0)
    plt.savefig("true_semi_loss.png")
    plt.close()

main_func()
