""" Class to record mouse-movement. """
# Author: Lukas Huber
# Created: 2021-11-25"
# Github: hubernikus

import logging
import warnings
import time
import datetime
import signal

import numpy as np

from abc import ABC, abstractmethod

# TODO: currently got an error installing pynput with python3.10
# from pynput import mouse
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


class BaseRecorder(ABC):
    def __init__(self, filename=None, sampling_time=0.1, max_it=10000):

        self.sampling_time = sampling_time

        if filename is None:
            now = datetime.datetime.now()
            self.filename = f"recorded_data/mousercording_{now:%Y-%m-%d_%H-%M-%S}.csv"

        else:
            filename = self.filename

        self.max_it = max_it
        self.simulation_stopped = True

    # def ctrl_c_handler(self, signum, frame):
    #     print("Ctrl-c was pressed. exiting")
    #     self.simulation_stopped = True

    def on_click(self, x, y, button, pressed):
        """Start stop recording toggle."""

        logging.info("Click event detected.")
        self.simulation_stopped = not (self.simulation_stopped)

        time.sleep(0.05)

    @abstractmethod
    def run(self) -> None:
        """Create a rund method which can use 'self.wait_for_sart'
        and self.store_to_file'"""
        pass

    def wait_for_click(self) -> np.ndarray:
        """Waits for simulation to start and return position."""
        self.dimension = 2
        positions = np.zeros((self.dimension, self.max_it + 2))

        it = 0
        while self.simulation_stopped:  #
            time.sleep(0.1)
            it += 1
            if not it % 10:
                logging.info("Waiting for click event to start.")

            if it > 100:
                raise TimeoutError("Recorder was not started for too long.")
        return positions

    def store_to_file(
        self, positions: np.ndarray, it_loop: int, it_traj: int = 0
    ) -> None:
        """Calculates position, velocity and acceleration."""

        if self.simulation_stopped:
            positions = positions[:, :it_loop]

        # Numerical derivative
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.sampling_time
        acceleration = (velocities[:, 1:] - velocities[:, :-1]) / self.sampling_time

        # Cut velocities
        velocities = 0.5 * velocities[:, 1:] + 0.5 * velocities[:, :-1]
        positions = (
            0.25 * positions[:, 2:]
            + 0.5 * positions[:, 1:-1]
            + 0.25 * positions[:, :-2]
        )

        # Save and evalute
        time_list = np.arange(0, positions.shape[1]) * self.sampling_time
        trajectory_id = it_traj * np.ones(positions.shape[1])

        # Store to csv
        logging.info(f"Storing to file: {self.filename}")

        if it_traj:  # not zero -> not the first one
            with open(self.filename, "a") as file:
                np.savetxt(
                    file,
                    np.vstack(
                        (trajectory_id, time_list, positions, velocities, acceleration)
                    ).T,
                    delimiter=",",
                )

        else:
            np.savetxt(
                self.filename,
                np.vstack(
                    (trajectory_id, time_list, positions, velocities, acceleration)
                ).T,
                delimiter=",",
                header=(
                    "trajectory_id, time [s], position_x, position_y, velocity_x, velocity_y, "
                    + "acceleration_x, acceleration_y"
                ),
            )


class MouseDataRecorder(BaseRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.listener = mouse.Listener(
            on_click=self.on_click,
            # on_scroll=on_scroll
        )

        # Mouse controller to access directly the position
        self._controller = mouse.Controller()

    def __del__(self):
        # Just in case -- not really needed I think.
        self.listener.stop()

    def run(self):
        self.listener.start()
        positions = self.wait_for_start()

        logging.info("Start recording.")
        for ii in range(self.max_it):
            if self.simulation_stopped:
                break

            time.sleep(self.sampling_time)

            positions[:, ii] = self._controller.position

        self.listener.stop()
        logging.info("Recording stopped with {max_it-2} data points.")

        self.store_to_file(positions, ii)


class MatplotlibMouseRecorder(BaseRecorder):
    def __init__(
        self, x_lim=None, y_lim=None, figsize=(12, 8), fig_ax_tuple=None, **kwargs
    ):
        super().__init__(**kwargs)

        if x_lim is None:
            x_lim = [0, 8]

        if y_lim is None:
            y_lim = [0, 6]

        plt.close("all")
        plt.ion()

        if fig_ax_tuple is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = fig_ax_tuple

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)

        plt.show()

        self.binding_id = plt.connect("motion_notify_event", self.on_move)
        plt.connect("button_press_event", self.on_click)

        self.mouse_coordinates = None

        self.it_trajectory = 0

    def on_move(self, event):
        # get the x and y pixel coords
        x, y = event.x, event.y

        if event.inaxes:
            # self.ax = event.inaxes  # the axes instance
            self.mouse_coordinates = np.array([event.xdata, event.ydata])
            print("data coords %f %f" % (event.xdata, event.ydata))

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            logging.info("Click event detected.")
            self.simulation_stopped = not (self.simulation_stopped)

    def wait_for_click(self) -> np.ndarray:
        """Waits for simulation to start and return position."""
        self.dimension = 2
        positions = np.zeros((self.dimension, self.max_it + 2))

        it = 0
        while self.simulation_stopped:
            plt.pause(0.1)
            it += 1
            if not it % 10:
                logging.info("Waiting for click event to start.")

            if it > 100:
                raise TimeoutError("Recorder was not started for too long.")

            if not plt.fignum_exists(self.fig.number):
                return None
        return positions

    def run(self):
        while plt.fignum_exists(self.fig.number):
            positions = self.wait_for_click()
            if positions is None:
                return

            # records, = self.ax.scatter([], [])
            (records,) = self.ax.plot(
                [],
                [],
                label=f"Recording #{self.it_trajectory}",
                # color="black",
                ms=10,
                marker="o",
                ls="",
            )
            self.ax.legend()
            logging.info("Start recording.")
            for ii in range(self.max_it):
                if not plt.fignum_exists(self.fig.number):
                    warnings.warn(
                        "Figures was closed during recording -- no saving executed."
                    )
                    return

                if self.simulation_stopped:
                    break

                plt.pause(self.sampling_time)

                if self.mouse_coordinates is None:
                    logging.info("Mouse has not been on the figure yet.")
                    continue

                positions[:, ii] = self.mouse_coordinates
                records.set_data(positions[0, : ii + 1], positions[1, : ii + 1])

            logging.info("Recording stopped with {max_it-2} data points.")
            self.store_to_file(positions, ii, self.it_trajectory)

            self.it_trajectory += 1


def _test_plotting_of_data():
    file_name = "mousercording_2022-09-16_12-24-13.csv"

    # Skip header
    data = np.loadtxt("recorded_data" + "/" + file_name, dtype="float", skiprows=1)

    breakpoint()


if (__name__) == "__main__":
    # TODO:
    # - command-line input
    # - Allow for several runs / recordings
    # - Visualize data using matplotlib
    logging.basicConfig(level=logging.INFO)

    # my_recorder = MouseDataRecorder()

    my_recorder = MatplotlibMouseRecorder()
    logging.info("Recorder is initialized...S tart / Stop on mouse-click.")
    my_recorder.run()

    # _test_plotting_of_data()

    print("Execution of file successful.")
