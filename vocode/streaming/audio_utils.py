import numpy as np


def int2float(sound):
    """
    Convert an array of integers representing sound samples to an array of floating point numbers.

    Parameters:
    sound (np.ndarray): The input array of sound samples.

    Returns:
    np.ndarray: The array of sound samples converted to floating point numbers.
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound
