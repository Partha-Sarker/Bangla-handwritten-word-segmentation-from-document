import cv2
import os
from pathlib import Path
from os.path import basename
import essentials as e
import sys

x0 = 0.9658246656760773
x1 = 2.389572773352643
y0 = 40.0
y1 = 20.0
increase_contour_m = (y0 - y1) / (x0 - x1)
increase_contour_c = y0 - increase_contour_m * x0

x0 = 6.76388840863001
x1 = 13.265119830389493
y0 = 81
y1 = 41
blur_h_kernel_m = (y0 - y1) / (x0 - x1)
blur_h_kernel_c = y0 - blur_h_kernel_m * x0

DOCUMENT_WIDTH = 1000
LINE_HEIGHT = 150


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_file_name_from_path(path):
    file_name_with_extension = os.path.basename(path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    return file_name


def get_parent_directory_from_path(path):
    path = Path(path)
    return basename(path.parent)


def get_line_percentage(contours):
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    line_count = len(contours)
    upper_contour = contours[0]
    lower_contour = contours[-1]
    min_point = upper_contour.min(0).flatten()[1]
    max_point = lower_contour.max(0).flatten()[1]
    percentage = line_count / (max_point - min_point) * 100
    return percentage


def get_contour_increase_amount(percentage):
    increase_size_amount = increase_contour_m * percentage + increase_contour_c
    increase_size_amount = round(increase_size_amount)
    if increase_size_amount < 15:
        increase_size_amount = 15
    elif increase_size_amount > 55:
        increase_size_amount = 55
    return increase_size_amount


def get_blur_h_kernel(percentage):
    blur_h_kernel = blur_h_kernel_m * percentage + blur_h_kernel_c
    blur_h_kernel = round(blur_h_kernel)

    if blur_h_kernel < 41:
        blur_h_kernel = 41
    elif blur_h_kernel > 83:
        blur_h_kernel = 83
    if blur_h_kernel % 2 == 0:
        blur_h_kernel += 1
    return blur_h_kernel


def segment_lines(src):
    if not os.path.exists(src):
        return
    image_name = get_file_name_from_path(src)
    original_image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    image = e.resize_image(original_image, width=DOCUMENT_WIDTH)

    th = e.convert_to_binary(image, 15, 15)
    blur = cv2.GaussianBlur(th, (199, 17), 57)

    blur = e.convert_to_binary(blur, 91, 3)
    blur_border = e.add_white_border(blur)

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(blur_border), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    larger_contours = e.get_larger_contours(contours, 1000)
    larger_contours = sorted(larger_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    line_percentage = get_line_percentage(larger_contours)
    increase_size_amount = get_contour_increase_amount(line_percentage)

    out_dir_name = 'segmented_lines/' + image_name + '/'
    create_directory(out_dir_name)

    line_number = 0
    line_locations = []
    black_percentage = 0
    for cnt in larger_contours:
        line_number += 1
        mask = e.get_contour_content(cnt, blur_border)
        expanded_mask = e.expand_mask(mask, h_kernel=increase_size_amount)
        expanded_line = e.get_mask_content(expanded_mask, th)
        expanded_cropped_mask = e.get_mask_content(expanded_mask, expanded_mask)
        line_file_location = '{}line_{}.jpg'.format(out_dir_name, line_number)
        cv2.imwrite(line_file_location, expanded_line)
        line_locations.append(line_file_location)
        black_percentage += e.get_black_percentage_from_mask(expanded_cropped_mask, expanded_line)
    black_percentage = float(black_percentage) / float(line_number)
    return line_locations, black_percentage


def segment_words(src, percentage):
    image_name = get_parent_directory_from_path(src)
    line_name = get_file_name_from_path(src)
    original_line = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    line = e.resize_image(original_line, height=LINE_HEIGHT)

    h_kernel = get_blur_h_kernel(percentage)

    blur_line = cv2.GaussianBlur(line, (h_kernel, 115), 0)
    blur_line = e.convert_to_binary(blur_line, 215, 9)

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(blur_line), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    larger_contours = e.get_larger_contours(contours, 200)
    larger_contours = sorted(larger_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    dir_name = 'segmented_words/' + image_name + '/'
    create_directory(dir_name)
    word_number = 0
    for cnt in larger_contours:
        word_number += 1
        mask = e.get_contour_content(cnt, blur_line)
        expanded_mask = e.expand_mask(mask, 15, 5)
        word = e.get_mask_content(expanded_mask, line)
        file_name = '{}{}_word_{}.jpg'.format(dir_name, line_name, word_number)
        cv2.imwrite(file_name, word)


def main(src):
    if not os.path.exists(src):
        print('file does not exist')
        return
    line_locations, black_percentage = segment_lines(src)
    image_name = get_file_name_from_path(src)
    print("All lines has been segmented from", image_name)
    for location in line_locations:
        segment_words(location, black_percentage)
    print("All words has been segmented from", image_name)

src = 'pages/test.jpg'
if len(sys.argv) == 2:
    src = sys.argv[1]
main(src)