import numpy as np
from sys import float_info as fi


def silence(
    sample_rate: int, duration: float, epsilon: bool = False, dtype=np.float64
) -> np.ndarray:
    """
    Generates silence block with the specified parameters.

    Args:
        sample_rate (int): [sampling rate]
        duration (float): [Silence duration]
        epsilon (bool, optional): [use the shorter value in the system instead of 0]. Defaults to False.

    Returns:
        [numpy.ndarray]: [Audio data signal]
    """
    output = np.zeros(shape=(int(np.ceil(duration * sample_rate)),), dtype=dtype)
    if epsilon:
        output = fi.epsilon * output
    return output


def delta(
    sample_rate: int, duration: float, amplitude: float, epsilon: bool = False
) -> np.ndarray:
    """
    Generates a delta dirac (impulse) with specific parameters.
    """
    output = silence(sample_rate, duration, epsilon=epsilon)
    output[0] = amplitude
    return output


if __name__ == "__main__":
    # generate a delta dirac with 1-second duration
    sample_rate = 44100
    duration = 1.0
    amplitude = 1.0
    delta_dirac = delta(sample_rate, duration, amplitude)
    print(f"delta_dirac: {delta_dirac[0:25]}")
