import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.signal import medfilt
from scipy.signal import savgol_filter


def signal_align(array1, array2, windows_size=100):
    """
    Compute the Pearson correlation coefficient within the specified region of interest,
    determine the offset between the two signals using weighted calculation,
    and eliminate amplitude differences between the signals to merge them into a single signal.

    :param array1: Signal 1 to be registered
    :param array2: Signal 2 to be registered
    :param windows_size: Initial value of the sliding window
    :return: Offset between signals, signal after registration
    """
    global concat_axis1, concat_axis2
    filtered_mode = 1
    k_size = 11

    # Preprocess the signal
    signal1, signal2 = signal_preprocessor(array1, array2, filtered_mode, k_size)

    array1_len = len(signal1)
    array2_len = len(signal2)
    max_value = 0

    # Generate zero arrays based on the length of the arrays to store the
    # results of correlation coefficient calculations.
    max_1 = np.zeros(array1_len)
    max_2 = np.zeros(array2_len)

    smaller_len = min(array1_len, array2_len)

    # Iterate to calculate the correlation coefficient.
    for i in range(windows_size, smaller_len):
        # Initialize the window for the computation.
        window1 = signal1[array1_len - i:]
        window2 = signal2[:i]

        # Calculate the Pearson correlation coefficient.
        correlation_coefficient, _ = pearsonr(window1, window2)
        max_1[i] = correlation_coefficient

    # Calculate the correlation coefficient by traversing in reverse.
    for j in range(windows_size, smaller_len):
        window1 = signal1[:j]
        window2 = signal2[array2_len - j:]

        correlation_coefficient, _ = pearsonr(window1, window2)
        max_2[j] = correlation_coefficient

    # Weighted calculation to find the optimal offset.
    s = find_weight_max(max_1, max_2)

    if s > 0:
        nan_array = np.full(s, np.nan)
        concat_axis1 = array1
        concat_axis2 = np.concatenate((nan_array, array2))
    else:
        nan_array = np.full(abs(s), np.nan)
        concat_axis1 = np.concatenate((nan_array, array1))
        concat_axis2 = array2



    return concat_axis1, concat_axis2, s


def signal_preprocessor(array1, array2, method, kernel_size):
    """
    :param array1:
    :param array2:
    :param method: Filtering methods
    :param kernel_size:
    :return:
    """
    array1_f = signal_filtered(array1, method, kernel_size)
    array2_f = signal_filtered(array2, method, kernel_size)

    # Normalize the signal.
    signal1 = min_max_normalization(array1_f)
    signal2 = min_max_normalization(array2_f)

    return signal1, signal2


def min_max_normalization(signal):
    """


    :param signal:
    :return:
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal


def signal_filtered(array, method, kernelsize):
    """
    Filter the signal according to the selection, currently only median filtering.

    :param array:
    :param method: Filtering methods
    :param kernelsize: kernel size
    :return: Filtered signal.
    """

    if method == 1:
        filtered_array = medfilt(array, kernel_size=kernelsize)
    elif method == 2:
        window = np.ones(kernelsize) / kernelsize

        filtered_array = np.convolve(array, window, mode='same')
    elif method == 3:
        window_size = 50
        filtered_array = np.convolve(array, np.ones(window_size) / window_size, mode='valid')
    elif method == 4:
        filtered_array = savgol_filter(array, window_length=11, polyorder=3)
    else:
        return array
    return filtered_array


def find_weight_max(array1, array2):
    array1_len = len(array1)
    array2_len = len(array2)

    weighted_1 = [val * 0.9 + i * 0.1 / len(array1) for i, val in enumerate(array1)]
    weighted_2 = [val * 0.9 + j * 0.1 / len(array2) for j, val in enumerate(array2)]

    max_index_1 = np.argmax(weighted_1)
    max_index_2 = np.argmax(weighted_2)

    if weighted_1[max_index_1] >= weighted_2[max_index_2]:
        return array1_len - max_index_1
    else:
        return max_index_2 - array2_len


if __name__ == "__main__":
    signal_1 = np.genfromtxt("signal1.csv", delimiter=',', encoding='utf-8-sig')
    signal_2 = np.genfromtxt("signal2.csv", delimiter=',', encoding='utf-8-sig')
    clean_signal_1 = signal_1[~np.isnan(signal_1)]
    clean_signal_2 = signal_2[~np.isnan(signal_2)]

    signal1, signal2, s = signal_align(clean_signal_1, clean_signal_2, int(len(clean_signal_1) / 5))
    print(s)

    plt.plot(signal1)
    plt.plot(signal2)
    plt.show()
