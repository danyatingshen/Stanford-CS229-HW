import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model

    theta_init = np.zeros((x_train.shape[1], 1))
    pr = PoissonRegression(step_size=lr, theta_0=theta_init, verbose=True)
    pr.fit(x_train, y_train.reshape(-1, 1))

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    pr.predict(x_eval)
    np.savetxt(save_path, pr.predict(x_eval))

    plt.figure()
    plt.scatter(y_eval, pr.predict(x_eval))
    plt.show()

    plot_x = np.linspace(0, len(pr.verbose), len(pr.verbose))
    plt.figure()
    plt.plot(plot_x, pr.verbose)
    plt.show()


# *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        verbose_temp = [] if self.verbose else self.verbose

        theta_old = np.copy(self.theta) + self.eps

        curr_iter = 0
        while curr_iter <= self.max_iter and np.max(np.abs(self.theta - theta_old)) >= self.eps:
            theta_old = np.copy(self.theta)

            self.theta = self.theta + self.step_size * x.T @ (y - self.predict(x))

            if self.verbose:
                verbose_temp.append(1/2 * np.sum(np.power(y - self.predict(x), 2)))

            curr_iter += 1

        self.verbose = verbose_temp
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
