""" Class which is used as basis for animations."""
# Author: Lukas Huber
# Date: 2021-11-25"
# Email: lukas.huber@epfl.ch
import os

from abc import ABC, abstractmethod
import datetime

import matplotlib.pyplot as plt
from matplotlib import animation

# from matplotlib.animation import PillowWriter


class Animator(ABC):
    """Abstract class to play animation and allow to save it.

    Properties
    ----------
    it_max: Maximum iterations of the animation (only visualization).
    dt_simulation: Time step of the
    dt_sleep: sleep time

    animation_name: The name the animation should be saved to.
    file_type: File type where animation is saved. Only used when no is given.

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


    Mouse/keyboard event
    --------------------
    MOUSE_CLICK / SPACE: Pause-play-toggle on click
    LEFT / 'a': One step back
    RIGHT / '': One step forward

    Trouble-Shooting
    ----------------
    When saving the animation, try to use python together with the animation,
    as in ipython environments the process is slower and has a (background)
    workspace which can lead to unexpected results.
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

    def step_forward(self) -> None:
        self.it_count += 1
        if self._animation_paused:
            self.update_step(self.it_count)

    def step_back(self) -> None:
        self.it_count -= 1
        if self._animation_paused:
            self.update_step(self.it_count)

    def pause_toggle(self, event=None) -> None:
        """Click event."""
        if self._animation_paused:
            self._animation_paused = False
        else:
            self._animation_paused = True

    def on_press(self, event):
        if event.key.isspace():
            self.pause_toggle()

        elif event.key == "right" or event.key == "d":
            self.step_forward()

        elif event.key == "left" or event.key == "a":
            self.step_back()

        # else:
        #    warnings.warn(f"Uknown key type {event}.")

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
        self.fig.canvas.mpl_connect("button_press_event", self.pause_toggle)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        if save_animation:
            # plt.rcParams["figure.autolayout"] = False
            plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

            if self.animation_name is None:
                now = datetime.datetime.now()
                animation_name = f"animation_{now:%Y-%m-%d_%H-%M-%S}" + self.file_type
            else:
                # Set filetype
                animation_name = self.animation_name + self.file_type

            print(f"Saving animation to: {animation_name}.")

            anim = animation.FuncAnimation(
                self.fig,
                self.update_step,
                frames=self.it_max,
                interval=self.dt_simulation * 1000,  # Conversion [s] -> [ms]
                blit=False,  # No optimization - but no return needed
            )

            # FFmpeg for
            # FFwriter = animation.FFMpegWriter(
            #     fps=30,
            #     # extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'],
            #     # extra_args=['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2'],
            #     # extra_args=['-vf', 'crop=1600:800']
            # )

            # FFwriter = animation.FFMpegWriter(fps=30, extra_args=["-vcodec", "h264"])
            # FFwriter = animation.FFMpegWriter(fps=30, extra_args=["-vcodec", "libx264"])

            # video_writer = animation.FFMpegWriter(
            #     fps=30,
            #     #     # extra_args=["-vcodec", "libx264"],
            #     #     # extra_args=["-vcodec", "h264"],
            # )

            video_writer = animation.PillowWriter(fps=round(1.0 / self.dt_simulation))

            anim.save(
                os.path.join("figures", animation_name),
                writer=video_writer,
                # fps=int(1.0 / self.dt_simulation),
                # metadata={"artist": "Lukas Huber"},
                # We chose default 'pillow', beacuse 'ffmpeg' often gives errors
                # writer=FFwriter,
            )
            print("Animation saving finished.")

        else:
            self.it_count = 0
            while self.it_max is None or self.it_count < self.it_max:
                if not plt.fignum_exists(self.fig.number):
                    print("Stopped animation on closing of the figure.")
                    break

                if self._animation_paused:
                    plt.pause(self.dt_sleep)
                    continue

                self.update_step(self.it_count)

                # Check convergence
                if self.has_converged(self.it_count):
                    print(f"All trajectories converged at it={self.it_count}.")
                    break

                # TODO: adapt dt_sleep based on
                plt.pause(self.dt_sleep)

                self.it_count += 1

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
        Returns boolean to indicate if the system has converged (to stop the
        simulation)."""
        return False

    def restore_figsize(self):
        """Reset to correct figure size before saving.
        Somehow three times fixes the size -> I'm really not sure why this hast
        to be done. but it overcomes the 'ffmpeg'-saving error.

        We are aware that this solution is very 'hacky', but somehow it solved
        the problem for now."""
        if not hasattr(self, "figsize"):
            self.figsize = self.fig.get_size_inches()

        self.fig.set_dpi(100)
        for _ in range(3):
            self.fig.set_size_inches(self.figsize[0], self.figsize[1], forward=True)
