""" Class which is used as basis for animations."""
# Author: Lukas Huber
# Date: 2021-11-25"
# Email: lukas.huber@epfl.ch

from abc import ABC, abstractmethod
import datetime

import numpy as np

from matplotlib import animation
import matplotlib.pyplot as plt


class Animator(ABC):
    """Abstract class to play animation and allow to save it.

    Properties
    ----------
    it_max: Maximum iterations of the animation (only visualization).
    dt_simulation: Time step of the
    dt_sleep: sleep time

    animation_name: The name the animation should be saved to.
    filetype:

    fig: 


    (Virtual) Methods -> implement for base-classes
    -----------------
    update_step: Is exectued at each iteration
    has_converged (optional): Returns bool to check if the system as converged.

    + an initialization (member) function is adviced; e.g. setup
    

    Methods 
    -------
    __init__: set the simulation paramters
    figure (or create_figure): assigns and returns matplotlib.pyplot.figure object
        this can also be assigned manually
    run: Run the simulation
        
    // Mouse/keyboard events:
    on_click: Pause/play on click
    
    
    """

    def __init__(
        self,
        it_max: int = 100,
        dt_simulation: float = 0.1,
        dt_sleep: float = 0.1,
        animation_name=None,
        file_type=".mp4",
    ):
        self.it_max = it_max

        self.dt_simulation = dt_simulation
        self.dt_sleep = dt_sleep

        self.animation_name = animation_name
        self.file_type = file_type

        # Simulation parameter
        self._animation_paused = False

    def on_click(self, event) -> None:
        # TODO: do space and forward/backwards event
        """Click event."""
        if self._animation_paused:
            self._animation_paused = False
        else:
            self._animation_paused = True

    def figure(self, *args, **kwargs) -> None:
        """ Creates a new figure and returns it."""
        self.fig = plt.figure(*args, **kwargs)
        return self.fig

    def create_figure(self, *args, **kwargs):
        return self.figure(*args, **kwargs)

    def run(self, save_animation: bool = False) -> None:
        """ Runs the animation"""
        if fig is None:
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
            while ii < it_max:
                if not plt.fignum_exists(self.fig.number):
                    print("Stopped animation on closing of the figure.")
                    break

                if self._animation_paused:
                    plt.pause(self.dt_sleep)
                    continue

                self.update_step(ii, animation_run=False)

                # Check convergence
                if self.has_converged():
                    print(f"All trajectories converged at it={ii}.")
                    break

                # TODO: adapt dt_sleep based on
                plt.pause(self.dt_sleep)

                ii += 1

    def has_converged(self) -> bool:
        """(Optional) convergence check which is called during each animation run."""
        return False

    @abstractmethod
    def update_step(self, ii: int) -> None:
        """Things which need to be done (and plotted) during a run.
        Ideally this should be structured as:

        1. calucaltion
        2. self.ax.clear()
        3. plotting
        """
        pass
