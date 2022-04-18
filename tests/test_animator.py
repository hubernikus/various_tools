#!/USSR/bin/python3.9
""" Test the animator class. """
# Author: Lukas Huber
# Created: 2022-05-06
# Email: lukas.huber@EFL.ch
# License: BSD (c) 2021

import numpy as np
import matplotlib.pyplot as plt

from vartools.animator import Animator


class AnimatorTest(Animator):
    """Test Animator which plots an evolving sinus curve."""

    def setup(self):
        self.x_width = 2 * np.pi
        self.y_lim = [-1.1, 1.1]

        self.fig, self.ax = plt.subplots(figsize=(6, 5))

        self.x_list = []
        self.y_list = []

    def update_step(self, ii):
        self.x_list.append(ii * 0.1)

        self.y_list.append(
            # The main function
            np.sin(ii * np.pi / 10)
        )

        while self.x_list[0] < self.x_list[-1] - self.x_width / 2:
            del self.x_list[0]
            del self.y_list[0]

        # Clear axes and plot
        self.ax.clear()
        self.ax.plot(self.x_list, self.y_list, color="blue")
        self.ax.plot(self.x_list[-1], self.y_list[-1], "o", color="blue")

        self.ax.set_xlim(self.x_list[0], self.x_list[0] + self.x_width)
        self.ax.set_ylim(self.y_lim)

    def has_converged(self, ii):
        return ii > 100


def test_animator(it_max=3):
    my_animator = AnimatorTest(dt_sleep=0.001, it_max=it_max)
    my_animator.setup()
    my_animator.run()

    plt.close("all")


def _test_saving():
    my_animator = AnimatorTest(dt_sleep=0.001, it_max=100)
    my_animator.setup()
    my_animator.run(save_animation=True)

    plt.close("all")


if (__name__) == "__main__":
    test_animator(it_max=10000)
    # _test_saving()
