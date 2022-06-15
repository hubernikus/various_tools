#!/USSR/bin/python3
"""Obstacle Avoidance Algorithm script with vecotr field. """
# Author: Lukas Huber
# Created: 2022-04-19
# Email: lukas.huber@epfl.ch

import os
import warnings
import datetime
from timeit import default_timer as timer

import numpy as np

# from numpy import linalg as LA

import matplotlib.pyplot as plt


class VectorfieldPlotter:
    def __init__(
        self,
        figsize=None,
        fig=None,
        ax=None,
        x_lim=None,
        y_lim=None,
        attractor_position=None,
    ):
        self.figsize = figsize

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        else:
            self.fig = fig
            self.ax = ax

        if x_lim is None:
            self.x_lim = [-10, 10]
        else:
            self.x_lim = x_lim

        if y_lim is None:
            self.y_lim = [-10, 10]
        else:
            self.y_lim = y_lim

        self.attractor_position = attractor_position

        # General Setup
        self.set_equal = True
        self.show_ticks = False
        self.default_label = False

        # Visualization parameters
        self.plottype = "streamplot"
        # self.plottype = "quiver"
        self.vector_color = "blue"
        self.vector_zorder = 0

        # Integration Parameters
        self.it_max = 1000
        self.dt_step = 0.1

        # Obstacle Plotting Parameters
        self.obstacle_color = np.array([176, 124, 124]) / 255.0

        self.plot_center = True
        self.plot_reference = False
        self.plot_obstacle_number = False

        self.velocity_arrow_factor = 0.2
        self.obstacle_alpha = 0.8
        self.obstacle_zorder = 0
        self.border_linestyle = "--"

        self.draw_wall_reference = False
        self.draw_velocity_arrow = False
        self.reference_point_number = False

        # Set evaluation points for debugging / close analysis
        self.positions = None

    def create_new_figure(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def setup_environment(self):
        if self.attractor_position is not None:
            self.ax.plot(
                self.attractor_position[0],
                self.attractor_position[1],
                "k*",
                linewidth=18.0,
                markersize=18,
                zorder=5,
            )

        if not self.show_ticks:
            self.ax.tick_params(
                axis="both",
                which="major",
                labelbottom=False,
                labelleft=False,
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

        if self.default_label:
            self.ax.set_xlabel(r"$\xi_1$")
            self.ax.set_ylabel(r"$\xi_2$")

        if self.set_equal:
            self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_xlim(self.x_lim[0], self.x_lim[1])
        self.ax.set_ylim(self.y_lim[0], self.y_lim[1])

    def plot(
        self, vector_functor, obstacle_list=None, check_functor=None, n_resolution=10
    ):
        if self.positions is None:
            nx = n_resolution
            ny = n_resolution
            x_vals, y_vals = np.meshgrid(
                np.linspace(self.x_lim[0], self.x_lim[1], nx),
                np.linspace(self.y_lim[0], self.y_lim[1], ny),
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

        else:
            positions = self.positions
            print(
                "[INFO] Plot based on existing positions, this only works"
                + "with quiver."
            )

        velocities = np.zeros(positions.shape)

        t_start = timer()
        if check_functor is not None:

            num_evaluations = 0
            for it in range(positions.shape[1]):
                if check_functor(positions[:, it]):
                    velocities[:, it] = vector_functor(positions[:, it])

                    num_evaluations += 1

        else:
            for it in range(positions.shape[1]):
                velocities[:, it] = vector_functor(positions[:, it])
            num_evaluations = positions.shape[1]
        t_end = timer()

        if not num_evaluations:  # zero points
            warnings.warn("No collision free points in space.")
        else:
            average_time = round((t_end - t_start) * 1000 / (num_evaluations), 3)
            print(f"Number of evaluations: {num_evaluations}")
            print(f"Average time per evaluation: {average_time} ms")

        if self.plottype == "quiver":
            self.ax.quiver(
                positions[0, :],
                positions[1, :],
                velocities[0, :],
                velocities[1, :],
                color=self.vector_color,
            )

        elif self.plottype == "streamplot":
            self.ax.streamplot(
                x_vals,
                y_vals,
                velocities[0, :].reshape(nx, ny),
                velocities[1, :].reshape(nx, ny),
                color=self.vector_color,
                zorder=self.vector_zorder,
            )

        else:
            raise ValueError(f"Unknown plottype '{plottype}'.")

        if obstacle_list is not None:
            self.plot_obstacles(obstacle_list)
        self.setup_environment()

    def plot_streamlines(
        self,
        positions,
        vector_functor,
        collision_functor=None,
        convergence_functor=None,
        obstacle_list=None,
    ):
        # traj_list = []
        for it_traj in range(positions.shape[1]):
            trajetory = np.zeros((positions.shape[0], self.it_max + 1))

            trajetory[:, 0] = positions[:, it_traj]
            for it_step in range(self.it_max):
                if collision_functor and collision_functor(trajetory[:, it_step]):
                    print(f"Trajectory {it_traj} collided at step {it_step}.")
                    # it_step -= 1
                    break

                if convergence_functor and convergence_functor(trajetory[:, it_step]):
                    print(f"Trajectory {it_traj} converged at step {it_step}.")
                    # it_step -= 1
                    break

                trajetory[:, it_step + 1] = trajetory[
                    :, it_step
                ] + self.dt_step * vector_functor(trajetory[:, it_step])

            self.ax.plot(
                trajetory[0, : it_step + 1],
                trajetory[1, : it_step + 1],
                color=self.vector_color,
            )
            # traj_list.append(trajetory[:, :(it_step+1)])

        # Do a few things
        if obstacle_list is not None:
            self.plot_obstacles(obstacle_list)
        self.setup_environment()

    def plot_obstacles(self, obstacle_container):
        obs_polygon = []
        obs_polygon_sf = []

        for n, obs in enumerate(obstacle_container):
            # Tiny bit outdated - newer obstacles will not need this check
            if hasattr(obs, "get_boundary_xy"):
                x_obs = np.array(obs.get_boundary_xy()).T

            else:
                # Outdated -> remove in the future
                obs.draw_obstacle()
                x_obs = obs.boundary_points_global_closed.T

            if hasattr(obs, "get_boundary_with_margin_xy"):
                x_obs_sf = np.array(obs.get_boundary_with_margin_xy()).T

            else:
                x_obs_sf = obs.boundary_points_margin_global_closed.T

            self.ax.plot(
                x_obs_sf[:, 0],
                x_obs_sf[:, 1],
                color="k",
                linestyle=self.border_linestyle,
            )

            if obs.is_boundary:
                if self.x_lim is None or self.y_lim is None:
                    raise Exception(
                        "Outer boundary can only be defined with `x_lim` and `y_lim`."
                    )
                outer_boundary = None
                if hasattr(obs, "global_outer_edge_points"):
                    outer_boundary = obs.global_outer_edge_points

                if outer_boundary is None:
                    outer_boundary = np.array(
                        [
                            [
                                self.x_lim[0],
                                self.x_lim[1],
                                self.x_lim[1],
                                self.x_lim[0],
                            ],
                            [
                                self.y_lim[0],
                                self.y_lim[0],
                                self.y_lim[1],
                                self.y_lim[1],
                            ],
                        ]
                    )

                outer_boundary = outer_boundary.T
                boundary_polygon = plt.Polygon(
                    outer_boundary, alpha=self.obstacle_alpha, zorder=-4
                )
                boundary_polygon.set_color(self.obstacle_color)
                self.ax.add_patch(boundary_polygon)

                obs_polygon.append(plt.Polygon(x_obs, alpha=1.0, zorder=-3))
                obs_polygon[n].set_color(np.array([1.0, 1.0, 1.0]))

            else:
                obs_polygon.append(
                    plt.Polygon(
                        x_obs, alpha=self.obstacle_alpha, zorder=self.obstacle_zorder
                    )
                )
                obs_polygon[n].set_color(self.obstacle_color)

            obs_polygon_sf.append(plt.Polygon(x_obs_sf, zorder=1, alpha=0.2))
            obs_polygon_sf[n].set_color([1, 1, 1])

            self.ax.add_patch(obs_polygon_sf[n])
            self.ax.add_patch(obs_polygon[n])

            if self.plot_obstacle_number:
                self.ax.annotate(
                    "{}".format(n + 1),
                    xy=np.array(obs.center_position) + 0.16,
                    textcoords="data",
                    size=16,
                    weight="bold",
                )

            if self.plot_center:
                self.ax.plot(
                    obs.center_position[0],
                    obs.center_position[1],
                    "k.",
                    # linewidth=12,
                    # markeredgewidth=2.4,
                    # markersize=8,
                )

            # Automatic adaptation of center
            if (
                self.plot_reference and not obs.is_boundary
            ) or self.draw_wall_reference:
                reference_point = obs.get_reference_point(in_global_frame=True)
                self.lax.plot(
                    reference_point[0],
                    reference_point[1],
                    "k+",
                    linewidth=12,
                    markeredgewidth=2.4,
                    markersize=8,
                )

            if self.reference_point_number:
                self.ax.annotate(
                    "{}".format(n),
                    xy=reference_point + 0.08,
                    textcoords="data",
                    size=16,
                    weight="bold",
                )

            if (
                self.draw_velocity_arrow
                and obs.linear_velocity is not None
                and np.linalg.norm(obs.linear_velocity) > 0
            ):
                velocity_color = [255 / 255.0, 51 / 255.0, 51 / 255.0]
                self.ax.arrow(
                    obs.center_position[0],
                    obs.center_position[1],
                    obs.linear_velocity[0] * self.velocity_arrow_factor,
                    obs.linear_velocity[1] * self.velocity_arrow_factor,
                    head_width=0.1,
                    head_length=0.1,
                    linewidth=3,
                    fc=velocity_color,
                    ec=velocity_color,
                    alpha=1,
                    zorder=3,
                )

    def evaluate_system(self, levelfunctor, n_resolution=100, obstacle_list=None):
        """Adds a very generic colorplot -> for more advanced features update."""
        nx = n_resolution
        ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(self.x_lim[0], self.x_lim[1], nx),
            np.linspace(self.y_lim[0], self.y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

        level_values = np.zeros(positions.shape[1])
        for ii in range(positions.shape[1]):
            level_values[ii] = levelfunctor(positions)

        return level_values

    def save(self, figurename=None, figtype="pdf", folder="figures"):
        if figurename is None:
            now = datetime.datetime.now()
            figurename = f"figure_{now:%Y-%m-%d_%H-%M-%S}"

        figurename = figurename + "." + figtype
        if folder is not None:
            figurename = os.path.join(folder, figurename)

        if self.fig is not None:
            self.fig.savefig(figurename, bbox_inches="tight")
        else:
            plt.savefig(figurename, bbox_inches="tight")
