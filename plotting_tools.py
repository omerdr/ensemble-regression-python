import matplotlib.pyplot as plt
from cycler import cycler
from numpy import mean

def plot_regression_results(ensemble, Z, y_test):
    # Plotting and saving to mat file
    color_cycler = cycler('c', 'rgbmy')
    marker_style_cycler = cycler('marker', 'xsovd.')
    style_cycler = marker_style_cycler * color_cycler * 2
    sty = iter(style_cycler)

    plt.figure(figsize=(15, 10))

    for i in range(ensemble.regressor_count):
        # s = sty.next()
        # zip the real and predicted values together, sort them, and unzip them
        points = zip(*sorted(zip(y_test, Z[i, :])))
        plt_x, plt_y = list(points[0]), list(points[1])
        # plt.scatter(boston_y_test, regr.predict(boston_X_test_scaled), label=str(regr), **sty.next())
        plt.plot(plt_x, plt_y, label=ensemble.regressor_labels[i], **sty.next())

    plt.plot(y_test, y_test, color='black', linewidth=2, label='truth', marker='d')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


def plot_y_e_correlation(ensemble, Z, y_test):
    plt.figure(figsize=(15, 10))
    color_cycler = cycler('c', 'rgbcmy')
    sty = iter(color_cycler)

    b_hat = mean(Z, 1) - mean(y_test)
    for i in range(ensemble.regressor_count):
        # s = sty.next()
        e = Z[i, :] - y_test - b_hat[i]
        plt.scatter(y_test, e, label=ensemble.regressor_labels[i], marker='x', **sty.next())

    plt.xlabel('True Response (y)')
    plt.ylabel('Error_i')
    plt.legend(bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()