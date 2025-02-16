#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    matches = []
    k, ell, c = template.shape
    scale_img = image
    res = cv2.matchTemplate(image,template,method=cv2.TM_CCORR_NORMED)
    loc = np.where(res >= detection_threshold)
    match = [(u, v, k, ell) for u, v in zip(*loc)]
    matches = matches + match
    for i in range(num_upscales):
        scale_img = cv2.pyrUp(scale_img)
        res = cv2.matchTemplate(scale_img,template,method=cv2.TM_CCORR_NORMED)
        loc = np.where(res >= detection_threshold)
        match = [(u//(2**(i+1)), v//(2**(i+1)), k//(2**(i+1)), ell//(2**(i+1))) for u, v in zip(*loc)]
        matches = matches + match
    scale_img = image
    for i in range(num_downscales):
        scale_img = cv2.pyrDown(scale_img)
        res = cv2.matchTemplate(scale_img,template,method=cv2.TM_CCORR_NORMED)
        loc = np.where(res >= detection_threshold)
        match = [(u*(2**(i+1)), v*(2**(i+1)), k*(2**(i+1)), ell*(2**(i+1))) for u, v in zip(*loc)]
        matches = matches + match
    return matches
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.97)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()
