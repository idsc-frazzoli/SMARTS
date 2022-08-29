from world import World, Resources, Player, Trajectory
# from env import MultiAgentEnv
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import os
import shutil
# import moviepy.video.io.ImageSequenceClip

from time import gmtime, strftime


def make_video(
        tmp_path,
        save_path,
        fps=20,
        remove_tmp: bool = True
):
    # import from smarts conda environment
    import moviepy.video.io.ImageSequenceClip
    image_files = [os.path.join(tmp_path, img)
                   for img in os.listdir(tmp_path)
                   if img.endswith(".png")]

    print(image_files)
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    print(save_path)
    clip.write_videofile(str(save_path), codec="libx264")

    if remove_tmp:
        shutil.rmtree(tmp_path)


def clear_cost(d):
    ds = 12.5
    cc = np.abs(ds-d)**3 if d < ds else 0
    return cc

if __name__ == "__main__":

    matplotlib.rcParams["text.usetex"] = True

    LARGESIZE = 16

    plt.rcParams.update({'font.size': LARGESIZE})
    plt.rcParams.update({'axes.titlesize': LARGESIZE})
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 18})
    plt.rcParams.update({'figure.titlesize': LARGESIZE})

    tts = 100
    traj_1 = Trajectory(total_time_steps=tts)
    traj_2 = Trajectory(total_time_steps=tts)

    x1 = [0] * tts
    x2 = [0] * tts
    y1 = [3] * tts
    # y1 = [10 - 13 / tts * x for x in range(tts)]
    y2 = [-3] * tts
    # y1 = [10 - 10 / tts * x for x in range(tts)]
    # y2 = [-10] * tts

    traj_1.set_from_lists(x1, y1)
    traj_2.set_from_lists(x2, y2)
    xlim, ylim = (-10, 10), (-10, 10)
    world = World(xlim, ylim, tts, 1)

    player_1 = Player(number=1, trajectory=traj_1, world=world)
    player_2 = Player(number=3, trajectory=traj_2, world=world)

    # set up resources
    dx, dy = 0.5, 0.5
    poly_cost = [1.0, 0.0, 0.0, 0.0]  # simple quadratic cost (e.g. [1, 0, 0] = 1*x^2 + 0*x^1 + 0*x^0)
    neigh_factors = [0.9, 0.4, 0.2]
    neigh_radii = [1.5, 3.0, 6.0]

    resources = Resources(world, dx, dy, poly_cost, neigh_factors, neigh_radii)

    cost = resources.calculate_cost([player_1, player_2])

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    ax1 = resources.plot(ax1)

    alphas = [0.5, 0.3, 0.15]
    ax1 = player_1.plot_used_resources(ax=ax1, k=tts // 2, alphas=alphas)
    ax1 = player_2.plot_used_resources(ax=ax1, k=tts // 2, alphas=alphas)

    # distance indicator
    lw = 2
    ax1.plot([-7, -7], [-3, 3], color="k", linewidth=lw)
    ax1.plot([-7.3, -6.7], [3, 3], color="k", linewidth=lw)
    ax1.plot([-7.3, -6.7], [-3, -3], color="k", linewidth=lw)
    ax1.text(-8.5, -.5, r"$d$", fontsize=28)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')

    #####################################################3
    tts = 100
    traj_1 = Trajectory(total_time_steps=tts)
    traj_2 = Trajectory(total_time_steps=tts)

    x1 = [0] * tts
    x2 = [0] * tts
    y1 = [10 - 13 / tts * x for x in range(tts)]
    y2 = [-3] * tts

    traj_1.set_from_lists(x1, y1)
    traj_2.set_from_lists(x2, y2)
    xlim, ylim = (-10, 10), (-10, 20)
    world = World(xlim, ylim, tts, 1)

    player_1 = Player(number=1, trajectory=traj_1, world=world)
    player_2 = Player(number=3, trajectory=traj_2, world=world)

    # set up resources
    dx, dy = 0.5, 0.5
    poly_cost = [1.0, 0.0, 0.0, 0.0]  # simple quadratic cost (e.g. [1, 0, 0] = 1*x^2 + 0*x^1 + 0*x^0)
    neigh_factors = [0.9, 0.4, 0.2]
    neigh_radii = [1.5, 3.0, 6.0]

    resources = Resources(world, dx, dy, poly_cost, neigh_factors, neigh_radii)

    cost = resources.calculate_cost([player_1, player_2])



    d = [y1[ts] - y2[ts] for ts in range(tts)]
    ax2.plot(d, cost[0], linewidth=3, color="#EB5353", label=r'$J_{i,t}^{\textrm{cg}}(d)$')
    ax2.set_ylim([0, max(cost[0]) + 100])
    ax2.grid()
    ax2.set_xlabel(r'$d$')
    ax2.set_ylabel(r'$\textrm{cost}$')

    cc = []
    for di in d:
        cc.append(1*clear_cost(di))
    ax2.plot(d, cc, linewidth=2, linestyle='--', color="k", label=r"$J_{i,t}^\textrm{clear}(d)$")

    ax2.legend()

    fig.savefig("test.png", dpi=400)
    fig.savefig("test.pdf")






# ######################################################################################################################
# if __name__ == "__main__":
#     total_time_steps = 100
#
#     # set up the two hard-coded trajectories
#     player_1_traj_close = Trajectory(total_time_steps=total_time_steps)
#     player_1_traj_far = Trajectory(total_time_steps=total_time_steps)
#     player_2_traj = Trajectory(total_time_steps=total_time_steps)
#
#     x1 = [2.5 * x * 20 / total_time_steps for x in range(total_time_steps)]
#     x2 = [x * 20 / total_time_steps + 15 for x in range(total_time_steps)]
#
#     y1_close = [3.5 * (np.sin(0.06 * x)) ** 3 - 3 for x in x1]
#     y1_far = [7 * (np.sin(0.06 * x)) ** 3 - 3 for x in x1]
#     y2 = [-2.5 for _ in range(total_time_steps)]
#
#     player_1_traj_close.set_from_lists(x1, y1_close)
#     player_1_traj_far.set_from_lists(x1, y1_far)
#     player_2_traj.set_from_lists(x2, y2)
#
#     # initialize world
#     xlim, ylim = (0, 50), (-5, 5)
#     world = World(xlim=xlim, ylim=ylim, total_time_steps=total_time_steps, n_lanes=2)
#
#     # set up players
#     player_1_close = Player(number=0, trajectory=player_1_traj_close, world=world)
#     player_1_far = Player(number=1, trajectory=player_1_traj_far, world=world)
#     player_2 = Player(number=2, trajectory=player_2_traj, world=world)
#
#     # set up resources
#     dx, dy = 1.0, 1.0
#     poly_cost = [1.0, 0.0, 0.0, 0.0]  # simple quadratic cost (e.g. [1, 0, 0] = 1*x^2 + 0*x^1 + 0*x^0)
#     neigh_factors = [1.0, 0.8, 0.5]
#     neigh_radii = [0.6, 3.0, 6.0]
#
#     resources = Resources(world, dx, dy, poly_cost, neigh_factors, neigh_radii)
#
#     cost_close = resources.calculate_cost([player_1_close, player_2])
#
#     cost_far = resources.calculate_cost([player_1_far, player_2])
#
#     social_cost_close = sum([sum(x) for x in cost_close])
#     social_cost_far = sum([sum(x) for x in cost_far])
#
#     print(social_cost_close)
#     print(social_cost_far)
#
#     matplotlib.rcParams.update({'font.size': 22})
#
#     # coeffs = np.arange(70, 100, 5)
#     # safety_dists = np.arange(3, 10, 1)
#     # k = 99
#     # for coeff in coeffs:
#     #     for sd in safety_dists:
#
#     for k in range(1, total_time_steps - 1):
#         #     print(k)
#         #  plot trajectories
#         fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(world.xlen / 2, 1 + 2 * world.ylen / 2), tight_layout=True)
#
#         ax1 = player_1_close.trajectory.plot(ax=ax1, world=world, color=player_1_close.color, k_range=[0, k + 1])
#         ax1 = player_2.trajectory.plot(ax=ax1, world=world, color=player_2.color, k_range=[0, k + 1])
#         # ax1 = player_1_far.trajectory.plot(ax=ax1, world=world, color=player_1_far.color, k_range=[0, k+1])
#
#         ax1 = resources.plot(ax1)
#
#         ax1 = player_1_close.plot_used_resources(ax=ax1, k=k)
#         # ax1 = player_1_far.plot_used_resources(ax=ax1, k=k)
#         ax1 = player_2.plot_used_resources(ax=ax1, k=k)
#
#         ax2.plot(cost_close[0][0:k], color="k", linewidth=2)
#         # ax2.plot(cost_far[0][0:k], color="k", linewidth=2)
#         ax2.scatter(total_time_steps, max(cost_close[0]), color="white")
#         ax2.scatter(0, 0, color="white")
#
#         ax2.set_xlabel('time step t')
#         ax2.set_ylabel('cost')
#         ax2.grid()
#
#         # clearance_cost_close = player_1_close.trajectory.clearance_cost(player_2.trajectory, coeff, sd)
#         # clearance_cost_far = player_1_far.trajectory.clearance_cost(player_2.trajectory, coeff, sd)
#         #
#         # ax2.plot(clearance_cost_close[0:k], color="red", linewidth=2)
#         # # ax2.plot(clearance_cost_far[0:k], color="red", linewidth=2)
#
#         ax1.set_aspect('equal', adjustable='box')
#
#         # hide x-axis
#         ax1.get_xaxis().set_visible(False)
#
#         # hide y-axis
#         ax1.get_yaxis().set_visible(False)
#
#         ax1.spines['top'].set_visible(False)
#         ax1.spines['right'].set_visible(False)
#         ax1.spines['bottom'].set_visible(False)
#         ax1.spines['left'].set_visible(False)
#
#         # fig.savefig("tmp/close_{}_{}.png".format(coeff, sd), dpi=100)
#         # # fig.savefig("tmp/far_{}.png".format(coeff), dpi=100)
#
#         fig.savefig("tmp/{:04d}.png".format(k), dpi=200)
#
#     make_video(Path("tmp"), "videos/close.mp4")
