import cv2
import numpy as np
import os


def resize_image(image, height=0, width=0):
    (h, w) = image.shape[:2]

    if height == 0 and width == 0:
        return image
    elif height != 0:
        r = height / float(h)
        dim = (int(w * r), height)
        resized = cv2.resize(image, dim)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        resized = cv2.resize(image, dim)

    return resized


def convert_to_binary(img, block_size=35, c=11):
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    #     th = cv2.bitwise_not(th)
    return th


def add_white_border(image, border_size=1, original_size=True):
    h, w = image.shape[0:2]

    if original_size:
        h_border = border_size
        w_border = border_size
        if h_border > h:
            h_border = h
        if w_border > w:
            w_border = w
        image_border = image.copy()
        image_border[:, 0:w_border] = 255
        image_border[:, -w_border:] = 255
        image_border[0:h_border, :] = 255
        image_border[-h_border:, :] = 255
    else:
        image_border = np.full([h + 2 * border_size, w + 2 * border_size], 255, np.uint8)
        image_border[border_size:-border_size, border_size:-border_size] = image
    return image_border


# def get_contour_content(contour, copy_from, original_size=True):
#     rect = cv2.boundingRect(contour)
#     x, y, w, h = rect
#     if original_size:
#         result = np.full_like(copy_from, 255)
#         result[y:y + h, x:x + w] = copy_from[y:y + h, x:x + w]
#     else:
#         result = np.empty([w, h], dtype=np.int8)
#         result = copy_from[y:y + h, x:x + w]
#     return result


# def get_mask_content(mask, source):
#     contours, hierarchy = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     largest_contour = get_largest_contour(contours)
#     word = get_contour_content(largest_contour, source, original_size=False)
#     return word


def get_contour_content(contour, copy_from, original_size=True):
    mask = np.full_like(copy_from, 255)
    cv2.fillPoly(mask, pts=[contour], color=0)
    mask = cv2.bitwise_not(mask)
    content = cv2.bitwise_and(mask, cv2.bitwise_not(copy_from))
    if not original_size:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        content = content[y:y + h, x:x + w]
    return cv2.bitwise_not(content)


def get_mask_content(mask, source):
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = get_largest_contour(contours)
    content = get_contour_content(largest_contour, source, False)
    return content


def expand_contours(contours, src_image, h_kernel=5, v_kernel=1, iterations=1):
    expanded_contours = []
    for c in contours:
        mask = get_contour_content(c, src_image)
        expanded_mask = expand_mask(mask, h_kernel, v_kernel, iterations)
        expanded_mask = add_white_border(expanded_mask, 1)
        expanded_contour, _ = cv2.findContours(cv2.bitwise_not(expanded_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        expanded_contour = get_largest_contour(expanded_contour)
        expanded_contours.append(expanded_contour)
    return expanded_contours


def expand_mask(mask, h_kernel=5, v_kernel=1, iterations=1):
    kernel = np.ones((h_kernel, v_kernel), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations)
    eroded = add_white_border(eroded)
    return eroded


def get_largest_contour(contours):
    largest_area = -3
    for contour in contours:
        if cv2.contourArea(contour) > largest_area:
            largest_contour = contour
    return largest_contour


def get_larger_contours(contours, min_area=50):
    cleaned_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cleaned_contours.append(c)
    return cleaned_contours
