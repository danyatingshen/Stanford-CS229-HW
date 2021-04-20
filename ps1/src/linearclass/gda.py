import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    util.plot(x_train, y_train, model.theta, 'result/p01e_{}.png'.format(save_path[-5]))
    # Use np.savetxt to save outputs from validation set to save_path
    inputs, labels = util.load_dataset(valid_path, add_intercept=True)
    predications = model.predict(inputs)
    np.savetxt(save_path, predications > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1.0, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n + 1)

        # Find phi, mu_0, mu_1, and sigma
        y_equal_1 = sum(y == 1)
        mu_0 = np.sum(x[y == 0], axis=0) / (m - y_equal_1)
        mu_1 = np.sum(x[y == 1], axis=0) / y_equal_1
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m
        phi = y_equal_1 / m
        # Write theta in terms of the parameters
        self.theta[0] = 0.5 * (mu_0 + mu_1).dot(np.linalg.inv(sigma)).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)
        self.theta[1:] = np.linalg.inv(sigma).dot(mu_1 - mu_0)
        
        # Return theta
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
