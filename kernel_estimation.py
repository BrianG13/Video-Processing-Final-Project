import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats.distributions import norm

NUM_OF_PLOTS = 5


def estimate_pdf(original_frame, indices,bw_method):
    omega_f_values = original_frame[indices[:, 0], indices[:, 1], :]
    pdf = gaussian_kde(omega_f_values.T,bw_method=bw_method)
    return lambda x: pdf(x.T)



    # import numpy as np
    # from scipy import stats
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # mu = np.array([1, 10, 20])
    # sigma = np.matrix([[4, 10, 0], [10, 25, 0], [0, 0, 100]])
    # data = np.random.multivariate_normal(mu, sigma, 1000)
    # values = data.T
    #
    # kde = stats.gaussian_kde(values)
    # density = kde(values)
    #
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # x, y, z = values
    # ax.scatter(x, y, z, c=density)
    # plt.show()
    return

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


if __name__ == "__main__":
    # The grid we'll use for plotting
    x_grid = np.linspace(-4.5, 3.5, 1000)

    # Draw points from a bimodal distribution in 1D
    np.random.seed(0)
    x = np.concatenate([norm(-1, 1.).rvs(400), norm(1, 0.3).rvs(100)])
    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))

    # Plot the three kernel density estimates
    fig, ax = plt.subplots(1, NUM_OF_PLOTS, sharey=True, figsize=(13, 3))
    fig.subplots_adjust(wspace=0)
    for i in range(NUM_OF_PLOTS):
        pdf = kde_scipy(x, x_grid, bandwidth=0.15 * (i + 1))
        ax[i].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
        ax[i].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
        ax[i].set_title('kde BW = {}'.format((i+1) * 0.15))
        ax[i].set_xlim(-4.5, 3.5)
    plt.show()
