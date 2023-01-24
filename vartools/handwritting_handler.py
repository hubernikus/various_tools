""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-05-10

import os
import logging

import numpy as np
from numpy import linalg as LA

from dataclasses import dataclass, field

import scipy
from scipy.io import loadmat

from vartools.directional_space import get_angle_space_of_array
from vartools.dynamical_systems import LinearSystem

Vector = np.ndarray
VectorArray = np.ndarray


@dataclass
class MotionDataHandler:
    """Stores (and imports) data for evaluation with the various learners.

    Attributes
    ----------
    positions: numpy-VectorArray of shape [n_datapoints x dimension]
    velocities: numpy-VectorArray of shape [n_datapoints x dimension]
    directions: numpy-VectorArray of shape [n_datapoints - 1 x dimension]
    time: numpy-Array of shape[n_datapoints]
    """

    position: VectorArray = field(default_factory=lambda: np.empty(0))
    velocity: VectorArray = field(default_factory=lambda: np.empty(0))
    sequence_value: VectorArray = field(default_factory=lambda: np.empty(0))

    direction: VectorArray = field(default_factory=lambda: np.empty(0))

    attractor_position: Vector = field(default_factory=lambda: np.empty(0))

    @property
    def attractor(self):
        return self.attractor_position

    @attractor.setter
    def attractor(self, value):
        self.attractor_position = value

    @property
    def num_samples(self) -> int:
        return self.position.shape[0]

    @property
    def n_samples(self) -> int:
        return self.position.shape[0]

    @property
    def dimension(self) -> int:
        return self.position.shape[1]

    # def normalize(self):
    #     self.mean_positions = np.mean(self.positions)
    #     self.var_positions = np.variance(self.positions)
    #     self.positions = (seplf.positions - self.mean_positions) / self.var_positions

    @property
    def X(self) -> VectorArray:
        return np.hstack(
            (self.position, self.velocity, self.sequence_value.reshape(-1, 1))
        )


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
            self.directory_name = "default/directory"
            self.directory_name = os.path.join(
                "/home", "lukas", "Code", "motion_learning_direction_space", "dataset"
            )
            # self.directory_name = os.path.join(
            #     "/home", "lukas", "Code", "lasahandwritingdataset", "DataSet"
            # )

        else:
            self.directory_name = directory_name
        self.file_name = file_name

        self.dimension = dimension

        # Define weights
        self.position_weight = 1
        self.direction_weight = 2
        self.sequence_weight = 3

        self.load_data_from_mat()

    @property
    def X(self) -> np.ndarray:
        return np.hstack(
            (
                self.position,
                self.velocity,
                self.direction.T * self.direction_weight,
                self.sequence_value.reshape(-1, 1) * self.sequence_weight,
            )
        )

    def get_normalized_data(self) -> np.ndarray:
        return np.hstack(
            (
                self.normalized_position * self.position_weight,
                self.normalized_velocity * self.direction_weight,
                self.sequence_value * self.sequence_weight,
            )
        )

    def load_data_from_mat(self, feat_in=None, attractor=None):
        """Load data from file mat-file & evaluate specific parameters"""
        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name)
        )

        if feat_in is None:
            self.feat_in = [0, 1]

        ii = 0  # Only take the first fold.

        # Normalize with std
        self.position = self.dataset["data"][0, ii][: self.dimension, :].T

        self.velocity = self.dataset["data"][0, ii][
            self.dimension : self.dimension * 2, :
        ].T

        self.sequence_value = np.linspace(0, 1, self.dataset["data"][0, ii].shape[1])
        self.start_positions = [self.dataset["data"][0, 0][:2, 0]]

        for it_set in range(1, self.dataset["data"].shape[1]):
            self.start_positions.append(self.dataset["data"][0, it_set][:2, 0])

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

        self.n_samples = self.position.shape[0]

        self.start_positions = np.array(self.start_positions).T
        self.sequence_value = self.sequence_value

        self.direction = get_angle_space_of_array(
            directions=self.velocity.T,
            positions=self.position.T,
            func_vel_default=LinearSystem(dimension=self.dimension).evaluate,
        )

        # Evaluate attractor position
        self.attractor_position = np.zeros((self.dimension))

        for it_set in range(0, self.dataset["data"].shape[1]):
            self.attractor_position = (
                self.attractor_position
                + self.dataset["data"][0, it_set][:2, -1].T
                / self.dataset["data"].shape[1]
            )

        # Normalize
        self.n_points = self.position.shape[1]

        self.mean_position = np.mean(self.position, axis=0)
        self.std_position = np.std(self.position, axis=0)

        self.normalized_position = (
            self.position - np.tile(self.mean_position, (self.n_samples, 1))
        ) / np.tile(self.std_position, (self.n_samples, 1))

        # We only care about the 'direction' of the velocity
        velocity_norm = LA.norm(self.velocity, axis=1)
        inds = velocity_norm > 0
        self.normalized_velocity = np.zeros_like(self.velocity)
        self.normalized_velocity[inds, :] = (
            self.velocity[inds, :] / np.tile(velocity_norm[inds], (self.dimension, 1)).T
        )
