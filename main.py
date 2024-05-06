import numpy as np
import Operation
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift


if __name__ == "__main__":
    signal_1 = np.genfromtxt('signal1.csv', delimiter=',', encoding='utf-8-sig')
    signal_2 = np.genfromtxt('signal2.csv', delimiter=',', encoding='utf-8-sig')

    #
    s, result = Operation.signal_align(signal_1, signal_2, 2)
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



