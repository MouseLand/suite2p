import numpy as np


def binMovie(ops):
    mov = np.zeros((ops['Ly'], ops['Lx'], ops['navg_frames_svd']), np.float32)

    with open(ops['reg_file'], 'r') as binary_file:
    # Read the whole file at once
    data = binary_file.read()
    print(data)
    return mov