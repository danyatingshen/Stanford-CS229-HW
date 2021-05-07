import numpy as np
import ast
import csv
import matplotlib.pyplot as plt

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader) # gets rid of the header
        rows = list(reader)
    inputs, labels = zip(*rows)
    inputs = np.array(list(map(from_np_array, inputs)), dtype=np.float)
    labels = np.array(labels, dtype=np.float)

    return inputs, labels

def plot(val_err, save_path, n_list, scale_list):
    """Plot dataset size vs. val err for different reg strengths

    Args:
        val_err: Matrix of validation errors.
        save_path: Path to save the plot.
        n_list: List of trainset sizes.
        scale_list: List of scaling for lambda.
    """
    # Plot dataset
    plt.figure()
    for i in range(len(scale_list)):
        plt.plot(n_list, val_err[i], linewidth=2, label=r'$\lambda=%0.4f\lambda_{opt}$'%scale_list[i])

    # Add labels and save to disk
    plt.xlabel('Num Samples')
    plt.ylabel('Validation Err')
    plt.ylim(0,2)
    plt.legend()
    plt.savefig(save_path)
