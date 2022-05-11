""" Class to record mouse-movement. """
# Author: Lukas Huber
# Created: 2021-11-25"
# Github: hubernikus

import logging
import time
import datetime

import numpy as np
from pynput import mouse


class MouseDataRecorder:
    def __init__(self, filename=None, sampling_time=0.01):
        self.sampling_time = sampling_time
        
        if filename is None:
            now = datetime.datetime.now()
            self.filename = f"mousercording_{now:%Y-%m-%d_%H-%M-%S}.csv"
            
        else:
            filename = self.filename
            
        self.listener = mouse.Listener(
            on_click=self.on_click,
            # on_scroll=on_scroll
        )

        self.simulation_stopped = True

        # Mouse controller to access directly the position
        self._controller = mouse.Controller()

    def __del__(self):
        # try:
        self.listener.stop()
        # except:
            # raise

    def on_click(self, x, y, button, pressed):
        """ Start stop recording toggle. """
            
        logging.info("Click event detected.")
        self.simulation_stopped = not(self.simulation_stopped)

        time.sleep(0.05)
        
        # if not self.simulation_stopped:
            # self.run()

    def run(self, max_it=10000):
        dimension = 2
        positions = np.zeros((dimension, max_it+2))

        self.listener.start()
        it = 0
        while self.simulation_stopped:
            time.sleep(0.1)
            it += 1
            if not it % 10:
                logging.info("Waiting for click event to start.")
                
            if it > 100:
                raise TimeoutError("Recorder was not started for too long.")
        print("here 70")

        logging.info("Start recording.")
        for ii in range(max_it):
            if self.simulation_stopped:
                break

            time.sleep(self.sampling_time)

            positions[:, ii] = self._controller.position
            
        self.listener.stop()
        logging.info("Recording stopped with {max_it-2} data points.")

        if self.simulation_stopped:
            positions = positions[:, :ii]

        # Numerical derivative
        velocities = (
            (positions[:, 1:] - positions[:, :-1]) / self.sampling_time
        )
        acceleration = (
            (velocities[:, 1:] - velocities[:, :-1]) / self.sampling_time
        )
        
        # Cut velocities
        velocities = 0.5 * velocities[:, 1:] + 0.5 * velocities[:, :-1]
        positions = (
            0.25 * positions[:, 2:]
            + 0.5 * positions[:, 1:-1]
            + 0.25 * positions[:, :-2]
        )

        # Save and evalute
        time_list = np.arange(0, positions.shape[1]) * self.sampling_time

        # Store to csv
        logging.info(f"Storing to file: {self.filename}")

        np.savetxt(
            self.filename,
            np.vstack((time_list, positions, velocities, acceleration)).T,
            delimiter=',',
            header=("time [s], position_x, position_y, velocity_x, velocity_y, "
                    + "acceleration_x, acceleration_y"),
        )


if (__name__) == "__main__":
    # TODO:
    # - command-line input
    # - Allow for several runs / recordings
    # - Visualize data using matplotlib
    logging.basicConfig(level=logging.INFO)

    my_recorder = MouseDataRecorder()
    logging.info("Recorder is initialized... Start / Stop on mouse-click.")
    my_recorder.run()
    
    print("done")

