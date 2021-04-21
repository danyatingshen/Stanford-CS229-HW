import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')


    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    model_true = LogisticRegression()
    model_true.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    util.plot(x_test, y_test, model_true.theta, 'plot_5a')

    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    np.savetxt(output_path_true, model_true.predict(x_test))

    # Part (b): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    model_naive = LogisticRegression()
    model_naive.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    util.plot(x_test, y_test, model_naive.theta, 'plot_5b')

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    np.savetxt(output_path_naive, model_naive.predict(x_test))

    # Part (f): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)

    x_index = np.where(y_valid == 1)

    alpha = 1 / len(y_valid[y_valid == 1]) * np.sum(model_naive.predict((x_valid[x_index])))

    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    util.plot(x_test, y_test, model_naive.theta, 'plot_5f', correction=alpha)

    np.savetxt(output_path_adjusted, model_naive.predict(x_test)*alpha)

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
