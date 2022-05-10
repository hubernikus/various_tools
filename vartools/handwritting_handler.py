""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-05-10

import os
import logging

from scipy.io import loadmat


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
        return self.data['demos'][0][0][0][0][0].shape[0]
    
    @property
    def dt(self):
        return self.data['dt'][0][0]

    @property
    def n_demonstrations(self):
        return self.data['demos'][0].shape[0]

    def get_demonstration(self, it_demo):
        return self.data['demos'][0][it_demo][0][0]

    def get_positions(self, it_demo):
        return self.data['demos'][0][it_demo][0][0][0]

    def get_times(self, it_demo):
        return self.data['demos'][0][it_demo][0][0][1]

    def get_velocities(self, it_demo):
        return self.data['demos'][0][it_demo][0][0][2]
    
    def get_accelerations(self, it_demo):
        return self.data['demos'][0][it_demo][0][0][3]

    def get_dt(self, it_demo=0):
        """ Returns the delta-time for a specific demo.
        A default argument is given as we assume to have the same dt for all demos. """
        return self.data['demos'][0][it_demo][0][0][4][0][0]
