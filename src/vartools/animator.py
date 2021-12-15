""" Class which is used as basis for animations."""
# Author: Lukas Huber
# Date: 2021-11-25
# Email: hubernikus@gmail.com

from abc import ABC, abstractmethod
import datetime

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


class Animator(ABC):
    """Abstract class to play animation and allow to save it.

    Properties
    ----------
    it_max: Maximum iterations of the animation (only visualization).
    dt_simulation: Time step of the
    dt_sleep: sleep time

    animation_name: The name the animation should be saved to.
    filetype: File type where animation is saved. Only used when no is given.

    fig: The figure object. Need for click etc. events

    Methods (Virtual)
    -----------------
    update_step: Update the simulation, but also the
    has_converged: Optional function to set a convergence check

        + an initialization (member) function is adviced; e.g. setup

    Methods
    -------
    __init__: set the simulation paramters
    figure (or create_figure): assigns and returns matplotlib.pyplot.figure object
        this can also be assigned manually
    subplots: use this suplots to create figure & axes which are further used in the
    update_step:

    run: Run the simulation

    // Mouse/keyboard events:
    on_click: Pause/play on click

    """

    def __init__(
        self,
        it_max: int = 100,
        iterator=None,  # Iterable
        dt_simulation: float = 0.1,
        dt_sleep: float = 0.1,
        animation_name=None,
        file_type=".mp4",
    ) -> None:
        self.it_max = it_max

        self.dt_simulation = dt_simulation
        self.dt_sleep = dt_sleep

        self.animation_name = animation_name
        self.file_type = file_type

        # Simulation parameter
        self._animation_paused = False

        # Additional arguments are passed to the custom-init
        # self._custom_init(*args, **kwargs)

    def on_click(self, event) -> None:
        """Click event."""
        if self._animation_paused:
            self._animation_paused = False
        else:
            self._animation_paused = True

    def figure(self, *args, **kwargs) -> None:
        """Creates a new figure and returns it."""
        self.fig = plt.figure(*args, **kwargs)
        return self.fig

    def create_figure(self, *args, **kwargs):
        return self.figure(*args, **kwargs)

    def run(self, save_animation: bool = False) -> None:
        """Runs the animation"""
        if self.fig is None:
            raise Exception("Member variable 'fig' is not defined.")

        # Initiate keyboard-actions
        cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        if save_animation:
            if self.animation_name is None:
                now = datetime.datetime.now()
                animation_name = f"animation_{now:%Y-%m-%d_%H-%M-%S}" + self.file_type
            else:
                # Set filetype
                animation_name = self.animation_name + self.file_type

            anim = animation.FuncAnimation(
                self.fig,
                self.update_step,
                frames=self.it_max,
                interval=self.dt_sleep * 1000,  # Conversion [s] -> [ms]
            )

            anim.save(
                os.path.join("figures", animation_name),
                metadata={"artist": "Lukas Huber"},
                # save_count=2,
            )
            print("Animation saving finished.")

        else:
            ii = 0
            while self.it_max is None or ii < self.it_max:
                if not plt.fignum_exists(self.fig.number):
                    print("Stopped animation on closing of the figure.")
                    break

                if self._animation_paused:
                    plt.pause(self.dt_sleep)
                    continue

                self.update_step(ii)

                # Check convergence
                if self.has_converged(ii):
                    print(f"All trajectories converged at it={ii}.")
                    break

                # TODO: adapt dt_sleep based on
                plt.pause(self.dt_sleep)

                ii += 1

    # @abstractmethod
    # def _custom_init(self, *args, **kwargs) -> None:
    # """Setup the environment including creating the axes."""
    # pass

    @abstractmethod
    def update_step(self, ii: int) -> None:
        """Things which need to be done (and plotted) during a run.
        Ideally this should be structured as:

        1. calucaltion
        2. self.ax.clear()
        3. plotting
        """
        pass

    def has_converged(self, ii: int) -> bool:
        """(Optional) convergence check which is called during each animation run.
        Returns boolean to indicate if the system has converged (to stop the simulation)."""
        return False
