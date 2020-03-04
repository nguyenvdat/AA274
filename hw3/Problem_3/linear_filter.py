#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    k, ell, c = F.shape
    m,n,c = I.shape
    G = np.zeros((m, n))
    F_vec = F.flatten()
    zero_width = (ell - n % ell) % ell
    zero_height = (k - m % k) % k
    I = np.hstack((I, np.zeros((m, zero_width, c))))
    I = np.vstack((I, np.zeros((zero_height, I.shape[1], c))))
    for i in range(I.shape[0] - k + 1):
        for j in range(I.shape[1] - ell + 1):
            I_vec = I[i:i+k, j:j+ell, :].flatten()
            G[i,j] = np.dot(F_vec, I_vec)
    return G
    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    k, ell, c = F.shape
    m,n,c = I.shape
    G = np.zeros((m, n))
    F_vec = F.flatten()
    zero_width = (ell - n % ell) % ell
    zero_height = (k - m % k) % k
    I = np.hstack((I, np.zeros((m, zero_width, c))))
    I = np.vstack((I, np.zeros((zero_height, I.shape[1], c))))
    for i in range(I.shape[0] - k + 1):
        for j in range(I.shape[1] - ell + 1):
            I_vec = I[i:i+k, j:j+ell, :].flatten()
            G[i,j] = np.dot(F_vec, I_vec) / (np.linalg.norm(F_vec)*np.linalg.norm(I_vec))
    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print('Correlation function runtime:', stop - start, 's')
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
