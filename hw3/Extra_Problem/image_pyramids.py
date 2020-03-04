#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    m, n, c = image.shape
    downscaled_row = np.zeros((0,n,c))
    for i in range(m):
        if i % 2 == 0:
            downscaled_row = np.vstack((downscaled_row, image[None,i,:,:]))
    downscaled_column = np.zeros((downscaled_row.shape[0],0,c))
    for i in range(downscaled_row.shape[1]):
        if i % 2 == 0:
            downscaled_column = np.hstack((downscaled_column, downscaled_row[:,None,i,:]))
    return downscaled_column
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.7)
    image = half_downscale(image)
    return image
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    image = np.repeat(image, 2, axis=0)
    image = np.repeat(image, 2, axis=1)
    return image
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    print(filt)
    ########## Code starts here ##########
    desired_img = np.zeros(((m-1)*scale+1, (n-1)*scale+1, c))
    for i in range(m-1):
        for j in range(n-1):
            desired_img[scale*i, scale*j, :] = image[i,j,:]
    desired_img = cv2.filter2D(desired_img, -1, filt)
    return desired_img
    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########
    # image = half_downscale(test_card)
    # image = half_downscale(image)
    # image = half_downscale(image)
    fig, ax = plt.subplots()
    # im = ax.imshow(image)
    # image = blur_half_downscale(test_card)
    # image = blur_half_downscale(image)
    # image = blur_half_downscale(image)
    # im = ax.imshow(image)
    # image = two_upscale(favicon)
    # image = two_upscale(image)
    # image = two_upscale(image)
    image = bilinterp_upscale(favicon, 3)
    im = ax.imshow(image)
    plt.show()
    ########## Code ends here ##########


if __name__ == '__main__':
    main()
