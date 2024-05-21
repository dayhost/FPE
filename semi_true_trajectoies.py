import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import constants
from fplanck import fokker_planck, boundary, gaussian_pdf
import matplotlib
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (6, 5)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4
import matplotlib.pylab as pl
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker

# 定义超参数
gamma = 0.9
r_s1_a1 = -0.1
r_s1_a2 = -0.1
phi_s1 = 0.1
phi_s2 = 0.1 / gamma - 0.05
phi_s3 = 0.1
phi_s4 = 0.1 / gamma + 0.05

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

    # 开始进行True Gradient Dynamics的轨迹可视化
    # 定义一个True Gradient的情景
    sample_rate = 500
    true_learning_rate = 0.1
    true_x_1, true_y_1 = generate_true_gradient_trajectory(
        start_point=[-2, 1], lr=true_learning_rate, step_n=50000)
    true_x_2, true_y_2 = generate_true_gradient_trajectory(
        start_point=[-2, 3], lr=true_learning_rate, step_n=50000)

    fig = plt.figure()
    # fig.suptitle("True Gradient Potential with trajectories")

    theta_a1_list, theta_a2_list =true_gradient_fpe.grid[0], true_gradient_fpe.grid[1]
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
    ax.plot(true_x_1, true_y_1, zorder=3, color="red")
    ax.scatter(true_x_1[::sample_rate], true_y_1[::sample_rate], s=10, zorder=3, label="trajectory 1", color="red")

    ax.plot(true_x_2, true_y_2, zorder=3, color="blue")
    ax.scatter(true_x_2[::sample_rate], true_y_2[::sample_rate], s=10, zorder=3, label="trajectory 2", color="blue")

    ax.scatter([solution_theta1_a1], [solution_theta1_a2], s=200, marker="*", zorder=2, color="C0")
    ax.scatter([solution_theta2_a1], [solution_theta2_a2], s=200, marker="*", zorder=2, color="C1")
    min_level = np.min(-np.log(true_stationary_dist))
    max_level = np.max(-np.log(true_stationary_dist))
    contour = ax.contour(theta_a1_list, theta_a2_list, -np.log(true_stationary_dist), colors="gray",
                         levels=np.arange(min_level, max_level + step_level, step_level), zorder=1)
    # ax.clabel(contour, colors="k")
    ax.set(xlabel=r'$\theta(a_1)$', ylabel=r'$\theta(a_2)$')
    # ax.set_title(r"$\sigma=$" + str(np.round(wanted_diffusion_constant, 4)) + " lr=" + str(true_learning_rate))
    # font = font_manager.FontProperties(size=14)
    plt.legend(loc=4)
    plt.tight_layout(pad=0)
    plt.subplots_adjust()
    plt.savefig("true_trajectory.png")
    plt.close()

    # 开始进行Semi Gradient Dynamics的轨迹可视化
    # 定义一个Semi Gradient的情景
    sample_rate = 100
    semi_learning_rate = 0.1
    semi_x_1, semi_y_1 = generate_semi_gradient_trajectory(
        start_point=[-2, 1], lr=semi_learning_rate, step_n=10000)
    semi_x_2, semi_y_2 = generate_semi_gradient_trajectory(
        start_point=[-2, 3], lr=semi_learning_rate, step_n=1700)

    fig = plt.figure()
    # fig.suptitle("Semi Gradient Potential with trajectories")

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
    ax.plot(semi_x_1, semi_y_1, zorder=3, color="red")
    ax.scatter(semi_x_1[::sample_rate], semi_y_1[::sample_rate], s=10, zorder=3, label="trajectory 1", color="red")

    ax.plot(semi_x_2, semi_y_2, zorder=3, color="blue")
    ax.scatter(semi_x_2[::sample_rate], semi_y_2[::sample_rate], s=10, zorder=3, label="trajectory 2", color="blue")

    ax.scatter([solution_theta1_a1], [solution_theta1_a2], s=200, marker="*", zorder=2, color="C0")
    ax.scatter([solution_theta2_a1], [solution_theta2_a2], s=200, marker="*", zorder=2, color="C1")
    min_level = np.min(-np.log(semi_stationary_dist))
    max_level = np.max(-np.log(semi_stationary_dist))
    contour = ax.contour(theta_a1_list, theta_a2_list, -np.log(semi_stationary_dist), colors="gray",
                         levels=np.arange(min_level, max_level + step_level, step_level), zorder=1)
    # ax.clabel(contour, colors="k")
    ax.set(xlabel=r'$\theta(a_1)$', ylabel=r'$\theta(a_2)$')
    # ax.set_title(r"$\sigma=$" + str(np.round(wanted_diffusion_constant, 4)) + " lr=" + str(semi_learning_rate))
    ax.legend(loc=4)

    plt.tight_layout(pad=0)
    plt.subplots_adjust()
    plt.savefig("semi_trajectory.png")
    plt.close()

main_func()
