import numpy as np


def filter_moving_average(data: np.ndarray, box_points: int = 1) -> np.ndarray:
    if len(data.shape) > 1:
        data_filtered = np.zeros(data.shape)
        for ii in range(data.shape[-1]):
            # TODO: not implemented for tensors (!)
            data_filtered[:, ii] = filter_moving_average(
                data[:, ii], box_points=box_points
            )
            return data_filtered

    filter_box = np.ones(box_points) / box_points
    data_filtered = np.convolve(data, filter_box, mode="same")
    return data_filtered
