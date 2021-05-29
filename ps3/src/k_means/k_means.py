from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# import cv2 if you cannot load image with mpimg (a bug found on Windows systems)
#import cv2

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    rand_indx_1 = np.random.randint(image.shape[0], size=num_clusters)
    rand_indx_2 = np.random.randint(image.shape[1], size=num_clusters)
    centroids_init = image[rand_indx_1, rand_indx_2, :]
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    temp_image = image.reshape(-1, 3)

    dist = [0]*centroids.shape[0]
    c = [0]*temp_image.shape[0]
    c_old = c.copy()

    new_centroids = np.copy(centroids)
    new_centroids = new_centroids.astype(float)

    for z in range(max_iter):
        for i in range(temp_image.shape[0]):
            pixel = temp_image[i, :]
            for j, centroid in enumerate(new_centroids):
                dist[j] = sum(np.subtract(pixel.astype(float), centroid ) ** 2)
            c[i] = dist.index(min(dist))

        for j, centroid in enumerate(new_centroids):
            new_centroids[j, :] = (temp_image[np.array(c) == j]).mean(axis=0).astype(float)

        if np.array_equal(np.asarray(c_old), np.asarray(c)):
            print("CONVERGENCE IN: " + str(z) + " ITERATIONS")
            break
        c_old = c.copy()

    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    original_shape = image.shape
    new_image = image.reshape(-1, 3).copy()

    dist = [0]*centroids.shape[0]

    for i in range(new_image.shape[0]):
        pixel = new_image[i, :]
        for j, centroid in enumerate(centroids):
            dist[j] = sum(np.subtract(pixel.astype(float), centroid) ** 2)
        new_image[i, :] = centroids[dist.index(min(dist)), :].astype(int)

    new_image = new_image.reshape(original_shape[0], original_shape[1], original_shape[2])
    # *** END YOUR CODE ***
    return new_image


def main(args):
    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    # Note: If you see error "TypeError: Image data of dtype object cannot be converted to float"
    # Comment off mpimg.imread() and uncomment the next line to use cv2.imread() instead
    # image = np.copy(cv2.cvtColor(cv2.imread(image_path_small), cv2.COLOR_BGR2RGB))

    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    # Note: If you see error "TypeError: Image data of dtype object cannot be converted to float"
    # Comment off mpimg.imread() and uncomment the next line to use cv2.imread() instead
    # image = np.copy(cv2.cvtColor(cv2.imread(image_path_large), cv2.COLOR_BGR2RGB))

    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
