import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(filename, load2=True, load3=True):
    """Loads data for 2's and 3's
    Inputs:
    filename: Name of the file.
    load2: If True, load data for 2's.
    load3: If True, load data for 3's.
    """
    assert (load2 or load3), "Atleast one dataset must be loaded."
    data = np.load(filename)
    if load2 and load3:
        inputs_train = np.hstack((data['train2'], data['train3']))
        inputs_valid = np.hstack((data['valid2'], data['valid3']))
        inputs_test = np.hstack((data['test2'], data['test3']))
        target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
        target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
        target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))
    else:
        
        if load2:
            inputs_train = data['train2']
            target_train = np.zeros((1, data['train2'].shape[1]))
            inputs_valid = data['valid2']
            target_valid = np.zeros((1, data['valid2'].shape[1]))
            inputs_test = data['test2']
            target_test = np.zeros((1, data['test2'].shape[1]))
        else:
            inputs_train = data['train3']
            target_train = np.zeros((1, data['train3'].shape[1]))
            inputs_valid = data['valid3']
            target_valid = np.zeros((1, data['valid3'].shape[1]))
            inputs_test = data['test3']
            target_test = np.zeros((1, data['test3'].shape[1]))

    return inputs_train.T, inputs_valid.T, inputs_test.T, target_train.T, target_valid.T, target_test.T


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax