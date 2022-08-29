import numpy as np
import math
import matplotlib.pyplot as plt

from typing import List

# AGENT_COLORS = {
#     0: '#FF0000',
#     1: '#00FF00',
#     2: '#0000FF',
#     3: '#FF00FF',
#     4: '#e41a1c'
# }

AGENT_COLORS = ["#F9D923", "#EB5353", "#00C897", "#2155CD"]


class Resources:
    def __init__(self,
                 world,
                 dx: float, dy: float,
                 poly_cost: List,
                 neigh_factors: List,
                 neigh_radii: List,
                 ):
        self.world = world
        self.dx, self.dy = dx, dy
        self.poly_cost = poly_cost
        self.neigh_factors = neigh_factors
        self.neigh_radii = neigh_radii
        self.n_neighs = len(neigh_factors)

        self.nx = int((self.world.xlim[1] - self.world.xlim[0]) // self.dx)
        self.ny = int((self.world.ylim[1] - self.world.ylim[0]) // self.dy)

    def get_position_by_index(self, ix: int, iy: int) -> List:
        x = self.world.xlim[0] + self.dx * ix + (self.dx / 2)
        y = self.world.ylim[1] - self.dy * iy - (self.dy / 2)
        return [x, y]

    def calculate_cost(self, players: List):

        # initialize 4D (x, y, time_step, neighborhood_degree) arrays for each player
        player_loads = np.zeros(shape=(len(players), self.n_neighs, self.world.total_time_steps, self.ny, self.nx))
        total_loads = np.zeros(shape=(self.n_neighs, self.world.total_time_steps, self.ny, self.nx))

        for i, player in enumerate(players):
            # print(i)
            for k in range(self.world.total_time_steps):
                player_pos = np.array([player.trajectory.x_pos[k], player.trajectory.y_pos[k]])
                for d in range(self.n_neighs):
                    for iy in range(self.ny):
                        for ix in range(self.nx):
                            resource_pos = np.array(self.get_position_by_index(ix, iy))
                            dist = np.linalg.norm(player_pos - resource_pos)
                            if dist < self.neigh_radii[d]:
                                player_loads[i, d, k, iy, ix] = 1
                                total_loads[d, k, iy, ix] += 1

        for i, player in enumerate(players):
            player.set_player_loads(player_loads[i, :, :, :, :])

        resource_costs = np.zeros(shape=(self.n_neighs, self.world.total_time_steps, self.ny, self.nx))

        for d in range(self.n_neighs):
            for k in range(self.world.total_time_steps):
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        for pci, factor in enumerate(self.poly_cost):
                            resource_costs[d, k, iy, ix] += self.neigh_factors[d] * factor * \
                                                            total_loads[d, k, iy, ix] ** (len(self.poly_cost) - 1 - pci)

        costs = [[0] * self.world.total_time_steps] * len(players)

        for i in range(len(players)):
            for k in range(self.world.total_time_steps):
                for d in range(self.n_neighs):
                    for iy in range(self.ny):
                        for ix in range(self.nx):
                            if player_loads[i, d, k, iy, ix]:
                                costs[i][k] += resource_costs[d, k, iy, ix]

        return costs

    def plot(self, ax):
        for x in range(self.nx + 1):
            ax.plot([self.world.xlim[0] + x * self.dx] * 2, [self.world.ylim[0], self.world.ylim[1]],
                    color='grey', linewidth=0.5)
        for y in range(self.ny + 1):
            ax.plot([self.world.xlim[0], self.world.xlim[1]], [self.world.ylim[0] + y * self.dy] * 2,
                    color='grey', linewidth=0.5)

        return ax


class World:
    def __init__(self,
                 xlim, ylim,
                 total_time_steps,
                 n_lanes
                 ):
        self.xlim = xlim
        self.ylim = ylim
        self.total_time_steps = total_time_steps
        self.n_lanes = n_lanes

        self.edge_markings, self.middle_markings = self.create_lane_markings()

        self.xlen = self.xlim[1] - self.xlim[0]
        self.ylen = self.ylim[1] - self.ylim[0]

    def create_lane_markings(self):
        edge_markings = [((self.xlim[0], self.ylim[1]), (self.xlim[1], self.ylim[1])),
                         ((self.xlim[0], self.ylim[0]), (self.xlim[1], self.ylim[0]))]
        middle_markings = []
        for mm in range(self.n_lanes - 1):
            h = (self.ylim[1] - self.ylim[0]) / self.n_lanes
            middle_markings.append(((self.xlim[0], self.ylim[1] - mm * h), (self.xlim[1], self.ylim[1] - mm * h)))

        return edge_markings, middle_markings

    def plot(self, ax):
        for em in self.edge_markings:
            ax.plot([em[0][0], em[1][0]], [em[0][1], em[1][1]], color='k', linewidth=0.1)
        for mm in self.middle_markings:
            ax.plot([mm[0][0], mm[1][0]], [mm[0][1], mm[1][1]], color='grey', linewidth=0.1, linestyle='--')

        return ax


class Trajectory:
    def __init__(self, total_time_steps):
        self.total_time_steps = total_time_steps
        self.time_steps = [t for t in range(total_time_steps)]
        self.x_pos = {t: 0.0 for t in self.time_steps}
        self.y_pos = {t: 0.0 for t in self.time_steps}
        self.x_list = None
        self.y_list = None

    def set_from_lists(self, x: List[float], y: List[float]):
        if len(x) != self.total_time_steps or len(y) != self.total_time_steps:
            print("Arrays must both have a length of {}".format(self.total_time_steps))

        self.x_list = x
        self.y_list = y

        for i in range(len(x)):
            self.x_pos[i] = x[i]
            self.y_pos[i] = y[i]

    def plot(self, world: World, ax, color, k_range):
        ax = world.plot(ax)

        ax.plot(self.x_list[k_range[0]:k_range[1]], self.y_list[k_range[0]:k_range[1]],
                color=color, linewidth=5)

        return ax

    def clearance_cost(self, traj2, coeff, safety_distance):
        if traj2.total_time_steps != self.total_time_steps:
            print("The trajectories must have the same length.")

        cost = []

        for i in range(self.total_time_steps):
            p1 = np.array([self.x_list[i], self.y_list[i]])
            p2 = np.array([traj2.x_list[i], traj2.y_list[i]])
            dist = np.linalg.norm(p1 - p2)
            cost.append(coeff * (safety_distance - dist) ** 2.0 if dist < safety_distance else 0.0)

        return cost


class Player:
    def __init__(self,
                 number,
                 trajectory,
                 world: World,
                 color=None):
        self.player_loads = None
        self.number = number
        self.trajectory = trajectory
        self.world = world
        if not color:
            self.color = AGENT_COLORS[self.number]
        else:
            self.color = color

    def set_player_loads(self, player_loads):
        self.player_loads = player_loads

    def plot_used_resources(self, ax, k: int, alphas: List = []):
        if not alphas:
            alphas = [1 / self.player_loads.shape[0] for _ in range(self.player_loads.shape[0])]
        dx, dy = self.world.xlen / self.player_loads.shape[3], self.world.ylen / self.player_loads.shape[2]
        for d in range(self.player_loads.shape[0]):
            for iy in range(self.player_loads.shape[2]):
                for ix in range(self.player_loads.shape[3]):
                    if self.player_loads[d, k, iy, ix]:
                        x_corners = [self.world.xlim[0] + ix * dx, self.world.xlim[0] + ix * dx + dx,
                                     self.world.xlim[0] + ix * dx + dx, self.world.xlim[0] + ix * dx]
                        y_corners = [self.world.ylim[1] - iy * dy, self.world.ylim[1] - iy * dy,
                                     self.world.ylim[1] - iy * dy - dy, self.world.ylim[1] - iy * dy - dy]
                        ax.fill(x_corners, y_corners, color=self.color, alpha=alphas[d])

        xpos, ypos = self.trajectory.x_pos[k], self.trajectory.y_pos[k]
        cx = [xpos - 0.9, xpos + 0.9, xpos + 0.9, xpos - 0.9]
        cy = [ypos + 0.45, ypos + 0.45, ypos - 0.45, ypos - 0.45]

        ax.fill(cx, cy, color="k", alpha=1)

        return ax
