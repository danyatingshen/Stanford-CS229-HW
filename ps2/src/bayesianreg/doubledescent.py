import numpy as np
import util

# Noise ~ N(0, sigma^2)
sigma = 0.5
# Dimension of x
d = 500
# Theta ~ N(0, eta^2*I)
eta = 1/np.sqrt(d)
# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
# List of dataset sizes
n_list = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]


def ridge_regression(train_path, validation_path):
    """
    For a specific training set, obtain theta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in scale_list.
    """
    # *** START CODE HERE ***
    train_data, train_labels = util.load_dataset(train_path)
    valid_data, valid_labels = util.load_dataset(validation_path)


    #eta_list = [ 1/np.sqrt(2 * scale_val) for scale_val in scale_list if scale_val != 0]
    #eta_list.insert(0, 10e-50)
    lam_opt = 1/(2*eta**2)
    val_err = []
    for lam in scale_list:
        theta_map = np.linalg.pinv(train_data.T @ train_data + 2*(sigma ** 2)*lam * np.eye(d) * lam_opt) @ train_data.T @ train_labels

        val_err.append( ((valid_data @ theta_map - valid_labels) ** 2).mean(axis=0))

    return val_err
    # *** END CODE HERE

if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    val_err = np.asarray(val_err).T
    util.plot(val_err, 'doubledescent.png', n_list, scale_list)
