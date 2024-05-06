from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt, find_peaks
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import bisect


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

    # Preprocess the signal and return it sorted by length.
    signal1, signal2, long_signal = signal_preprocessor(array1, array2, 1, 7)

    array1_len = len(signal1)
    array2_len = len(signal2)
    max_value = 0

    # Generate zero arrays based on the length of the arrays to store the
    # results of correlation coefficient calculations.
    max_1 = np.zeros(array1_len)
    max_2 = np.zeros(array2_len)

    # Iterate to calculate the correlation coefficient.
    for i in range(windows_size, array2_len):
        # Initialize the window for the computation.
        window1 = signal1[array1_len - i:]
        window2 = signal2[:i]

        # Calculate the Pearson correlation coefficient.
        correlation_coefficient, _ = pearsonr(window1, window2)
        max_1[i] = correlation_coefficient

        if correlation_coefficient > max_value:
            max_value = correlation_coefficient
            s = array1_len - i
            print(s)

    # Calculate the correlation coefficient by traversing in reverse.
    for i in range(windows_size, array2_len):
        window1 = signal1[:i]
        window2 = signal2[array2_len - i:]

        correlation_coefficient, _ = pearsonr(window1, window2)
        max_2[i] = correlation_coefficient

        if correlation_coefficient > max_value:
            max_value = correlation_coefficient
            s = i - array2_len
            print(s)

    # Weighted calculation to find the optimal offset.
    s = find_weight_max(max_1, max_2)

    # Reorder the original signal according to the signal length.
    if long_signal == 1:
        signal1 = signal_filtered(array1, 0, 7)
        signal2 = signal_filtered(array2, 0, 7)
    else:
        signal2 = signal_filtered(array1, 0, 7)
        signal1 = signal_filtered(array2, 0, 7)

    # Concatenate the signals based on the offset.
    result = registration(s, signal1, signal2)

    return abs(s), result

def signal_preprocessor(array1, array2, method, kernel_size):
    """
    :param array1:
    :param array2:
    :param method: Filtering methods
    :param kernel_size:
    :return: Filtered longer signal, filtered shorter signal, the numbering of longer signals in the original signal.
    """
    array1_len = len(array1)
    array2_len = len(array2)

    # Filtering the signal.
    array1_f = signal_filtered(array1, method, kernel_size)
    array2_f = signal_filtered(array2, method, kernel_size)

    # Sort the original signal by length.
    if array1_len >= array2_len:
        signal1 = array1_f
        signal2 = array2_f
        long_signal = 1
    else:
        signal1 = array2_f
        signal2 = array1_f
        long_signal = 2

    # Normalize the signal.
    data1_2d = signal1.reshape(-1, 1)
    data2_2d = signal2.reshape(-1, 1)
    scaler = MinMaxScaler()
    single1_n = scaler.fit_transform(data1_2d)
    single2_n = scaler.fit_transform(data2_2d)
    signal1 = single1_n.flatten()
    signal2 = single2_n.flatten()

    return signal1, signal2, long_signal


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
    else:
        return array
    return filtered_array


def registration(s, signal_1, signal_2):
    """
    After normalizing using Z-Score, perform inverse normalization to
    eliminate differences between duplicated signals.

    :param s: Offset between signals
    :param signal_1:
    :param signal_2:
    :return: Registered signal
    """
    if s >= 0:
        r1 = signal_1
        r2 = signal_2
    if s < 0:
        r1 = signal_2
        r2 = signal_1

    length_1 = len(r1)
    length_2 = len(r2)
    s = abs(s)

    normalized_data1 = (r1 - np.mean(r1[s:])) / np.std(r1[s:])
    normalized_data2 = (r2 - np.mean(r2[:length_1 - s])) / np.std(r2[:length_1 - s])

    avg_ratio = np.mean(r2[:length_1 - s]) / np.mean(r1[s:])
    print("avg_ratio", avg_ratio)

    if avg_ratio >= 1:
        unstandardized_data = normalized_data1 * np.std(r2[:length_1 - s]) \
                              + np.mean(r2[:length_1 - s])
        result = np.concatenate((unstandardized_data[:s - 1], r2[:]))
    else:
        unstandardized_data = normalized_data2 * np.std(r1[s:]) + np.mean(r1[s:])
        result = np.concatenate((r1, unstandardized_data[length_1 - s - 1:]))

    return result


def find_peaks_test(array1, array2):
    array_1_peaks, _ = find_peaks(array1)
    array_2_peaks, _ = find_peaks(array2)
    return array_1_peaks, array_2_peaks


def find_weight_max(array1, array2):
    array1_len = len(array1)
    array2_len = len(array2)

    weighted_1 = [val * 0.9 + i * 0.1 / len(array1) for i, val in enumerate(array1)]
    weighted_2 = [val * 0.9 + i * 0.1 / len(array2) for i, val in enumerate(array2)]

    max_index_1 = np.argmax(weighted_1)
    max_index_2 = np.argmax(weighted_2)

    if max_index_1 >= max_index_2:
        return array1_len - max_index_1
    else:
        return max_index_2 - array2_len


if __name__ == "__main__":
    signal_1 = np.genfromtxt('signal1.csv', delimiter=',', encoding='utf-8-sig')
    signal_2 = np.genfromtxt('signal2.csv', delimiter=',', encoding='utf-8-sig')

    #
    s, result = signal_align(signal_1, signal_2, 10)
    print('Shift value to align is', s)

    plt.subplot(2, 1, 1)
    plt.plot(signal_1, label='singal_1')
    plt.plot(signal_2, label='singal_2')
    plt.legend(loc='best')
    plt.ylim(bottom=0)

    plt.subplot(2, 1, 2)
    plt.plot(result, label='registration')
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.show()
