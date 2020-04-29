import matplotlib.pyplot as plt
import numpy as np

def plot_2dmatrix(data_matrix, ylabel="y"):
    print(data_matrix.shape)
    plt.plot(data_matrix)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    test_matrix = np.random.rand(2, 200)
    print(test_matrix.shape)
    plot_2dmatrix(test_matrix)