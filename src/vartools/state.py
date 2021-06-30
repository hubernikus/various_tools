'''
Basic stat to base objects on
'''
# Author: Lukas Huber
# Mail: lukas.huber@epfl.ch
# License: BSD (c) 2021

# import time
import copy

import numpy as np
import matplotlib.pyplot as plt    # Only for debugging


class State(object):
    """ Basic state class which allows encapsulates further.  """
    def __init__(self, typename=None, State=None, name="default", reference_frame="world"):
        if State is not None:
            self = copy.deepcopy(State)
        else:
            self.typename = typename
            self.reference_frame = reference_frame
            self.name = name
            
    @property
    def typename(self):
        return self._typename

    @typename.setter
    def typename(self, value):
        self._typename = value

    @property
    def reference_frame(self):
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        self._reference_frame = value

    # Multiplication and division not commutative
    # def __repr__(self):
        # return "State(%r, %r, %r, %r)" % (self._typename, self._name, self._reference_frame, self.)

    # def __str__(self):
        # if self._empty:
            # res = "Empty "
        # res += "State: " + self._name + " expressed in " + self._reference_frame + " frame"
        # return res

    # def __typename__(self):
        # return self.__repr__
