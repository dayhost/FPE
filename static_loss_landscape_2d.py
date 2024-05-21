import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from fplanck import fokker_planck, boundary, gaussian_pdf
import argparse
import matplotlib
matplotlib.rc('font', size=14)
matplotlib.rcParams['figure.figsize'] = (6, 5)
matplotlib.rcParams["legend.markerscale"] = 2
matplotlib.rcParams["legend.columnspacing"] = 1.5
matplotlib.rcParams["legend.labelspacing"] = 0.4


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


def main_func():

    if phi_s1 == 0.1:
        t1 = "s1"
    elif np.isclose(phi_s1, 0.1 / gamma - 0.05):
        t1 = "s2"
    elif np.isclose(phi_s1, 0.1 / gamma + 0.05):
        t1 = "s3"
    elif phi_s1 == 0:
        t1 = "s4"

    if phi_s2 == 0.1:
        t2 = "s1"
    elif np.isclose(phi_s2, 0.1 / gamma - 0.05):
        t2 = "s2"
    elif np.isclose(phi_s2, 0.1 / gamma + 0.05):
        t2 = "s3"
    elif phi_s2 == 0:
        t2 = "s4"

    if phi_s3 == 0.1:
        t3 = "s1"
    elif np.isclose(phi_s3, 0.1 / gamma - 0.05):
        t3 = "s2"
    elif np.isclose(phi_s3, 0.1 / gamma + 0.05):
        t3 = "s3"
    elif phi_s3 == 0:
        t3 = "s4"

    if phi_s4 == 0.1:
        t4 = "s1"
    elif np.isclose(phi_s4, 0.1 / gamma - 0.05):
        t4 = "s2"
    elif np.isclose(phi_s4, 0.1 / gamma + 0.05):
        t4 = "s3"
    elif phi_s4 == 0:
        t4 = "s4"

    nm = 1e-2  # 定义单位长度

    solution_pi1_a1 = r_s1_a1 / (phi_s1 - gamma * phi_s2)
    solution_pi1_a2 = r_s1_a2 / phi_s3 + gamma * (phi_s4 / phi_s3) * r_s1_a1 / (phi_s1 - gamma * phi_s2)

    solution_pi2_a1 = r_s1_a1 / phi_s1 + gamma * (phi_s2 / phi_s1) * r_s1_a2 / (phi_s3 - gamma * phi_s4)
    solution_pi2_a2 = r_s1_a2 / (phi_s3 - gamma * phi_s4)

    if solution_pi1_a1 > solution_pi1_a2 and solution_pi2_a1 > solution_pi2_a2:
        center = [solution_pi1_a1, solution_pi1_a2]
    elif solution_pi1_a1 < solution_pi1_a2 and solution_pi2_a1 < solution_pi2_a2:
        center = [solution_pi2_a1, solution_pi2_a2]
    else:
        center = [(solution_pi1_a1 + solution_pi2_a1) / 2, (solution_pi1_a2 + solution_pi2_a2) / 2]

    # center = [1 / 8, 5 / 4]
    # center = [0, 0]
    extent = [950 * nm, 950 * nm]
    resolution = 9.5 * nm

    init_pdf_center = (center[0], center[1])
    init_pdf_width = 30 * nm

    propagation_time = 100000

    sample_interval = 10
    step_level = 1

    if t1 == "s1" and t2 == "s2" and t3 == "s1" and t4 == "s3":
        true_vector_scale = 1e-3
        semi_vector_scale = 1e-3
    elif t1 == "s1" and t2 == "s2" and t3 == "s2" and t4 == "s4":
        true_vector_scale = 6e-4
        semi_vector_scale = 6e-4
    elif t1 == "s1" and t2 == "s2" and t3 == "s3" and t4 == "s4":
        true_vector_scale = 1e-3
        semi_vector_scale = 3e-3
    elif t1 == "s2" and t2 == "s1" and t3 == "s1" and t4 == "s3":
        true_vector_scale = 1e-3
        semi_vector_scale = 1e-3
    elif t1 == "s2" and t2 == "s1" and t3 == "s2" and t4 == "s4":
        true_vector_scale = 6e-4
        semi_vector_scale = 6e-4
    elif t1 == "s2" and t2 == "s1" and t3 == "s3" and t4 == "s4":
        true_vector_scale = 1e-3
        semi_vector_scale = 3e-3
    elif t1 == "s3" and t2 == "s1" and t3 == "s1" and t4 == "s3":
        true_vector_scale = 1e-3
        semi_vector_scale = 2e-3
    elif t1 == "s3" and t2 == "s1" and t3 == "s2" and t4 == "s4":
        true_vector_scale = 6e-4
        semi_vector_scale = 2e-3
    elif t1 == "s3" and t2 == "s1" and t3 == "s3" and t4 == "s4":
        true_vector_scale = 1e-3
        semi_vector_scale = 3e-3

    wanted_mobility_constant = 1  # 期待的Mobility Constant Coefficient
    # wanted_diffusion_constant_list = [2**-2, 2**-3, 2**-4, 2**-5]
    wanted_diffusion_constant = 2 ** -8
    boundary_condition = boundary.reflecting
    stationary_dist_minimal_value = 1e-40

    # Figure 2 (a)
    drag = 1 / wanted_mobility_constant  # drag = 1/mobility, 我们希望mobility为1
    temperature = wanted_diffusion_constant * drag / constants.k  # 定义 Diffusion = kT/drag，也就是 T = Diffusion * drag / k

    true_gradient_fpe = fokker_planck(temperature=temperature, drag=drag, extent=extent, center=center,
                resolution=resolution, boundary=boundary_condition, force=true_gradient_force)

    # 计算Exact True Loss Landscape
    grid = true_gradient_fpe.grid
    true_stationary_dist = np.zeros([len(grid[0]), len(grid[1])])
    for i in range(len(grid[0][0])):
        for j in range(len(grid[1][0])):
            x = grid[0][i][j]
            y = grid[1][i][j]

            U1 = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * x) ** 2 + 1 / 2 * (
                    phi_s3 * y - r_s1_a2 - gamma * phi_s4 * x) ** 2
            U2 = 1 / 2 * (phi_s1 * x - r_s1_a1 - gamma * phi_s2 * y) ** 2 + 1 / 2 * (
                    phi_s3 * y - r_s1_a2 - gamma * phi_s4 * y) ** 2

            if x >= y:
                true_stationary_dist[i][j] = np.exp(-U1 / wanted_diffusion_constant)
            else:
                true_stationary_dist[i][j] = np.exp(-U2 / wanted_diffusion_constant)

    # 找出Stationary Distribution里面哪里计算结果为0
    area_mask = generate_mask_by_stationary_distribution(stationary_distribution=true_stationary_dist)
    # 把计算为0的点填上最小值
    true_stationary_dist[true_stationary_dist == 0] = stationary_dist_minimal_value
    # 对于Stationary distribution进行归一化
    true_stationary_dist_volume = np.trapz(
        np.trapz(true_stationary_dist, true_gradient_fpe.grid[0][:, 0], axis=0),
        true_gradient_fpe.grid[1][0, :], axis=0)
    true_stationary_dist = true_stationary_dist / true_stationary_dist_volume
    # 计算flux和gradient
    theta_a1_list, theta_a2_list = true_gradient_fpe.grid[0], true_gradient_fpe.grid[1]
    true_force = true_gradient_force(theta_a1_list=theta_a1_list, theta_a2_list=theta_a2_list,
                                     sample_rate=1)
    true_flux = calculate_flux_of_coordinate(theta_a1_list=theta_a1_list, theta_a2_list=theta_a2_list,
                                             stationary_distribution=true_stationary_dist,
                                             sample_rate=1,
                                             force_x1=true_force[0], force_x2=true_force[1],
                                             wanted_diffusion_constant=wanted_diffusion_constant)
    true_gradient = calculate_gradient_from_stationary_distribution(theta_a1_list=theta_a1_list,
                                                                          theta_a2_list=theta_a2_list,
                                                                          stationary_distribution=true_stationary_dist,
                                                                          sample_rate=1)
    # 去掉那些stationary Distribution为0的点计算出来的gradient和flux，这是不准的噪声
    true_gradient = [np.multiply(true_gradient[0], area_mask), np.multiply(true_gradient[1], area_mask)]
    true_flux = [np.multiply(true_flux[0], area_mask), np.multiply(true_flux[1], area_mask)]
    true_force = [np.multiply(true_force[0], area_mask), np.multiply(true_force[1], area_mask)]

    # 使用stationary Distribution来归一化flux和gradient，使其之和与force相等
    true_gradient, true_flux = divide_force_into_flux_and_gradient(
        stationary_distribution=true_stationary_dist, flux=true_flux, gradient=true_gradient,
        wanted_diffusion_constant=wanted_diffusion_constant, sample_rate=1)

    # 去掉那些太长的vector
    vector_length_mask = generate_mask_by_vector_length(gradient_x=true_gradient[0], gradient_y=true_gradient[1],
                                                        flux_x=true_flux[0], flux_y=true_flux[1],
                                                        vector_scale=true_vector_scale)
    true_gradient = [np.multiply(true_gradient[0], vector_length_mask), np.multiply(true_gradient[1], vector_length_mask)]
    true_flux = [np.multiply(true_flux[0], vector_length_mask), np.multiply(true_flux[1], vector_length_mask)]
    true_force = [np.multiply(true_force[0], vector_length_mask), np.multiply(true_force[1], vector_length_mask)]

    # 采样
    true_force = [true_force[0][::sample_interval, ::sample_interval],
                  true_force[1][::sample_interval, ::sample_interval]]
    true_gradient = [true_gradient[0][::sample_interval, ::sample_interval],
                     true_gradient[1][::sample_interval, ::sample_interval]]
    true_flux = [true_flux[0][::sample_interval, ::sample_interval],
                 true_flux[1][::sample_interval, ::sample_interval]]

    print("finish calculate distribution")

    # 开始进行画图
    fig = plt.figure()

    theta_a1_list, theta_a2_list = true_gradient_fpe.grid[0], true_gradient_fpe.grid[1]
    sampled_theta_a1_list = theta_a1_list[::sample_interval, ::sample_interval]
    sampled_theta_a2_list = theta_a2_list[::sample_interval, ::sample_interval]
    ax = fig.add_subplot(1, 1, 1)
    # 根据超参数画出critical point
    if solution_pi1_a1 > solution_pi1_a2:
        ax.scatter([solution_pi1_a1], [solution_pi1_a2], s=200, marker="*", zorder=3, color="C1")

    if solution_pi2_a1 < solution_pi2_a2:
        ax.scatter([solution_pi2_a1], [solution_pi2_a2], s=200, marker="*", zorder=3, color="C0")
    # 画出policy变化的虚线
    policy_line_theta_1 = []
    policy_line_theta_2 = []
    for theta_1 in theta_a1_list[:, 0]:
        if theta_1 < min(theta_a2_list[0, :]) or theta_1 > max(theta_a2_list[0, :]):
            continue
        policy_line_theta_1.append(theta_1)
        policy_line_theta_2.append(theta_1)
    ax.plot(policy_line_theta_1, policy_line_theta_2, linestyle="dashed", color="k", label="policy boundary", zorder=3)
    # 画出等高线图
    min_level = np.min(-np.log(true_stationary_dist))
    max_level = np.max(-np.log(true_stationary_dist))
    contour = ax.contour(theta_a1_list, theta_a2_list,
                         -np.log(true_stationary_dist), colors="gray",
                         levels=np.arange(min_level, max_level + step_level, step_level), zorder=1)
    # ax.clabel(contour, colors="k", levels=np.arange(min_level, max_level + 3 * step_level, 3 * step_level))
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , true_force[0],
              true_force[1], units='dots', color="red", label="force", scale=true_vector_scale, zorder=2)
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , true_flux[0],
              true_flux[1], units='dots', color="green", label="flux", scale=true_vector_scale, zorder=2)
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , true_gradient[0],
              true_gradient[1], units='dots', color="orange", label="gradient", scale=true_vector_scale, zorder=2)
    ax.set(xlabel=r'$\theta(a_1)$', ylabel=r'$\theta(a_2)$')

    ax.legend(loc=4)
    plt.subplots_adjust()
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.tight_layout(pad=0)
    plt.savefig("true_loss_landscape_three_vector_"+ t1 + "_a1_" + t2 + "_" + t3 + "_a2_" + t4 + ".png")
    plt.close()

    # figure 2 (b)
    semi_gradient_fpe = fokker_planck(temperature=temperature, drag=drag, extent=extent, center=center,
                resolution=resolution, boundary=boundary.reflecting, force=semi_gradient_force)

    # 进行单个时刻的运算
    semi_gradient_init_pdf = gaussian_pdf(center=init_pdf_center, width=init_pdf_width)
    semi_stationary_dist = semi_gradient_fpe.propagate(semi_gradient_init_pdf, time=propagation_time)
    area_mask = generate_mask_by_stationary_distribution(stationary_distribution=semi_stationary_dist)
    semi_stationary_dist[semi_stationary_dist == 0] = stationary_dist_minimal_value

    semi_stationary_dist_volume = np.trapz(
        np.trapz(semi_stationary_dist, semi_gradient_fpe.grid[0][:, 0], axis=0),
        semi_gradient_fpe.grid[1][0, :], axis=0)
    semi_stationary_dist = semi_stationary_dist / semi_stationary_dist_volume

    theta_a1_list, theta_a2_list = semi_gradient_fpe.grid[0], semi_gradient_fpe.grid[1]
    semi_force = semi_gradient_force(theta_a1_list=theta_a1_list, theta_a2_list=theta_a2_list, sample_rate=1)
    semi_flux = calculate_flux_of_coordinate(theta_a1_list=theta_a1_list, theta_a2_list=theta_a2_list,
                                             stationary_distribution=semi_stationary_dist,
                                             sample_rate=1,
                                             force_x1=semi_force[0], force_x2=semi_force[1],
                                             wanted_diffusion_constant=wanted_diffusion_constant)
    semi_gradient = calculate_gradient_from_stationary_distribution(theta_a1_list=theta_a1_list,
                                                                    theta_a2_list=theta_a2_list,
                                                                    stationary_distribution=semi_stationary_dist,
                                                                    sample_rate=1)

    semi_gradient = [np.multiply(semi_gradient[0], area_mask), np.multiply(semi_gradient[1], area_mask)]
    semi_flux = [np.multiply(semi_flux[0], area_mask), np.multiply(semi_flux[1], area_mask)]
    semi_force = [np.multiply(semi_force[0], area_mask), np.multiply(semi_force[1], area_mask)]

    semi_gradient, semi_flux = divide_force_into_flux_and_gradient(
        stationary_distribution=semi_stationary_dist, flux=semi_flux, gradient=semi_gradient,
        wanted_diffusion_constant=wanted_diffusion_constant, sample_rate=1)

    semi_force = [semi_force[0][::sample_interval, ::sample_interval],
                  semi_force[1][::sample_interval, ::sample_interval]]
    semi_gradient = [semi_gradient[0][::sample_interval, ::sample_interval],
                     semi_gradient[1][::sample_interval, ::sample_interval]]
    semi_flux = [semi_flux[0][::sample_interval, ::sample_interval],
                 semi_flux[1][::sample_interval, ::sample_interval]]

    print("finish calculate distribution")

    # 开始进行画图
    fig = plt.figure()
    theta_a1_list, theta_a2_list = semi_gradient_fpe.grid[0], semi_gradient_fpe.grid[1]
    sampled_theta_a1_list = theta_a1_list[::sample_interval, ::sample_interval]
    sampled_theta_a2_list = theta_a2_list[::sample_interval, ::sample_interval]

    ax = fig.add_subplot(1, 1, 1)

    if solution_pi1_a1 > solution_pi1_a2:
        ax.scatter([solution_pi1_a1], [solution_pi1_a2], s=200, marker="*", zorder=3, color="C1")

    if solution_pi2_a1 < solution_pi2_a2:
        ax.scatter([solution_pi2_a1], [solution_pi2_a2], s=200, marker="*", zorder=3, color="C0")

    policy_line_theta_1 = []
    policy_line_theta_2 = []
    for theta_1 in theta_a1_list[:, 0]:
        if theta_1 < min(theta_a2_list[0, :]) or theta_1 > max(theta_a2_list[0, :]):
            continue
        policy_line_theta_1.append(theta_1)
        policy_line_theta_2.append(theta_1)
    ax.plot(policy_line_theta_1, policy_line_theta_2, linestyle="dashed", color="k", label="policy boundary", zorder=3)

    # ax.scatter([semi_critical_point_list[i][0] ], [semi_critical_point_list[i][1] ], s=100,
    #            marker="^", zorder=3)
    min_level = np.min(-np.log(semi_stationary_dist))
    max_level = np.max(-np.log(semi_stationary_dist))
    contour = ax.contour(theta_a1_list, theta_a2_list, -np.log(semi_stationary_dist), colors="gray",
                         levels=np.arange(min_level, max_level + step_level, step_level), zorder=1)
    # ax.clabel(contour, colors="k")
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , semi_force[0],
              semi_force[1], units='dots', color="red", label="force", scale=semi_vector_scale, zorder=2)
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , semi_flux[0],
              semi_flux[1], units='dots', color="green", label="flux", scale=semi_vector_scale, zorder=2)
    ax.quiver(sampled_theta_a1_list , sampled_theta_a2_list , semi_gradient[0],
              semi_gradient[1], units='dots', color="orange", label="gradient", scale=semi_vector_scale, zorder=2)
    ax.set(xlabel=r'$\theta(a_1)$', ylabel=r'$\theta(a_2)$')

    ax.legend()
    plt.tight_layout(pad=0)
    plt.subplots_adjust()
    plt.savefig("semi_loss_landscape_three_vector_"+ t1 + "_a1_" + t2 + "_" + t3 + "_a2_" + t4 + ".png")
    plt.close()


if __name__ == "__main__":
    # PARSER = argparse.ArgumentParser()
    # PARSER.add_argument("--phi-s1", type=float)
    # PARSER.add_argument("--phi-s2", type=float)
    # PARSER.add_argument("--phi-s3", type=float)
    # PARSER.add_argument("--phi-s4", type=float)
    # PARSER.add_argument("--gamma", type=float)
    # PARSER.add_argument("--r", type=float)
    #
    # ARGS = PARSER.parse_args()
    #
    # # 定义超参数
    # gamma = ARGS.gamma
    # r_s1_a1 = ARGS.r
    # r_s1_a2 = ARGS.r
    #
    # phi_s1 = ARGS.phi_s1
    # phi_s2 = ARGS.phi_s2
    # phi_s3 = ARGS.phi_s3
    # phi_s4 = ARGS.phi_s4

    # 定义超参数
    gamma = 0.9
    r_s1_a1 = -0.1
    r_s1_a2 = -0.1

    phi_s1 = 0.1 / gamma - 0.05
    phi_s2 = 0.1
    phi_s3 = 0.1
    phi_s4 = 0.1 / gamma + 0.05

    # phi_s1 = 0.1
    # phi_s2 = 0.1 / gamma - 0.05
    # phi_s3 = 0.1
    # phi_s4 = 0.1 / gamma + 0.05
    main_func()
