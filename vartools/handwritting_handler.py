""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-05-10

import os
import logging

import numpy as np

import scipy
from scipy.io import loadmat

from vartools.directional_space import get_angle_space_of_array
from vartools.dynamical_systems import LinearSystem


class HandwrittingDataHandler:
    def __init__(self, dataset_name, dataset_dir=None):
        # TODO: allow reading from online directly
        if dataset_dir is None:
            dataset_dir = "/home/lukas/Code/handwritting_dataset/DataSet"

        file_name = os.path.join(dataset_dir, dataset_name)
        self.data = loadmat(file_name)
        logging.info("Finished data loading.")

    @property
    def dimensions(self):
        return self.data["demos"][0][0][0][0][0].shape[0]

    @property
    def dt(self):
        return self.data["dt"][0][0]

    @property
    def n_demonstrations(self):
        return self.data["demos"][0].shape[0]

    def get_demonstration(self, it_demo):
        return self.data["demos"][0][it_demo][0][0]

    def get_positions(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][0]

    def get_times(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][1]

    def get_velocities(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][2]

    def get_accelerations(self, it_demo):
        return self.data["demos"][0][it_demo][0][0][3]

    def get_dt(self, it_demo=0):
        """Returns the delta-time for a specific demo.
        A default argument is given as we assume to have the same dt for all demos."""
        return self.data["demos"][0][it_demo][0][0][4][0][0]


class HandwrittingHandler:
    def __init__(self, file_name, directory_name: str = None, dimension: int = 2):
        if directory_name is None:
            # self.directory_name = "default/directory"
            self.directory_name = os.path.join(
                "/home", "lukas", "Code", "motion_learning_direction_space", "dataset"
            )
        else:
            self.directory_name = directory_name
        self.file_name = file_name

        self.dimension = dimension

        self.load_data_from_mat()

    def load_data_from_mat(self, feat_in=None, attractor=None):
        """Load data from file mat-file & evaluate specific parameters"""

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name)
        )

        if feat_in is None:
            self.feat_in = [0, 1]

        ii = 0  # Only take the first fold.
        self.position = self.dataset["data"][0, ii][: self.dimension, :].T
        self.velocity = self.dataset["data"][0, ii][
            self.dimension : self.dimension * 2, :
        ].T

        self.sequence_value = np.linspace(0, 1, self.dataset["data"][0, ii].shape[1])

        for it_set in range(1, self.dataset["data"].shape[1]):
            self.position = np.vstack(
                (self.position, self.dataset["data"][0, it_set][:2, :].T)
            )
            self.velocity = np.vstack(
                (self.velocity, self.dataset["data"][0, it_set][2:4, :].T)
            )

            # TODO include velocity - rectify
            self.sequence_value = np.hstack(
                (
                    self.sequence_value,
                    np.linspace(0, 1, self.dataset["data"][0, it_set].shape[1]),
                )
            )

        direction = get_angle_space_of_array(
            directions=self.velocity.T,
            positions=self.position.T,
            func_vel_default=LinearSystem(dimension=self.dimension).evaluate,
        )

        self.X = np.hstack((self.position, self.velocity, direction.T))

        # self.X = self.normalize_velocity(self.X)

        self.num_samples = self.X.shape[0]
        self.dim_gmm = self.X.shape[1]

        weightDir = 4

        if attractor is None:
            self.attractor = np.zeros((self.dimension))

            for it_set in range(0, self.dataset["data"].shape[1]):
                self.attractor = (
                    self.attractor
                    + self.dataset["data"][0, it_set][:2, -1].T
                    / self.dataset["data"].shape[1]
                )

        # Normalize dataset
        normalize_dataset = False
        if normalize_dataset:
            self.meanX = np.mean(self.X, axis=0)

            self.meanX = np.zeros(4)
            self.varX = np.var(self.X, axis=0)

            # All distances should have same variance
            self.varX[: self.dim] = np.mean(self.varX[: self.dim])

            # All directions should have same variance
            self.varX[self.dim : 2 * self.dim - 1] = np.mean(
                self.varX[self.dim : 2 * self.dim - 1]
            )

            # Stronger weight on directions!
            self.varX[self.dim : 2 * self.dim - 1] = (
                self.varX[self.dim : 2 * self.dim - 1] * 1 / weightDir
            )

            self.X = self.X / np.tile(self.varX, (self.X.shape[0], 1))

        else:
            self.meanX = None
            self.varX = None
