import cv2
import numpy as np
import os
import essentials as e
import sys
from pathlib import Path

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_file_name_from_path(path):
    file_name_with_extension = os.path.basename(path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    return file_name

def get_parent_directory_from_path(path):
    path = Path(path)
    return str(path.parent)

def segment_words_from_line(src):
    image_name = get_parent_directory_from_path(src)
    line_name = get_file_name_from_path(src)
    original_line = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    
    line = e.resize_image(original_line, height=LINE_HEIGHT)
    blur_line = cv2.GaussianBlur(line, (19, 19), 0)
    blur_line = e.convert_to_binary(blur_line, 91, 22)
    
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(blur_line), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    reduced_contours = e.get_larger_contours(contours)
    
    dir_name = 'segmented_words/'+image_name+'/'
    create_directory(dir_name)
    word_number = 0
    for cnt in reduced_contours:
        word_number += 1
        mask = e.get_contour_content(cnt, blur_line)
        expanded_mask = e.expand_mask(mask, 10, 4, 1)
        word = e.get_mask_content(expanded_mask, line)
        file_name = '{}{}_word_{}.jpg'.format(dir_name, line_name, word_number)
        cv2.imwrite(file_name, word)

def segment_lines_from_document(src):
    image_name = get_file_name_from_path(src)
    original_image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    image = e.resize_image(original_image, width=PAGE_WIDTH)
    
    th = e.convert_to_binary(image, 35, 11)
    blur = cv2.GaussianBlur(th, (75, 13), 15)
    blur = e.convert_to_binary(blur, 33, 3)
    blur_border = e.add_white_border(blur,1)

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(blur_border), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    larger_contours = e.get_larger_contours(contours, 50)
    larger_contours = sorted(larger_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    #expanded_contours = e.expand_contours(larger_contours, blur_border, 10, 1, 1)
    out_dir_name = 'segmented_lines/'+image_name+'/'
    create_directory(out_dir_name)
    
    line_locations = []
    line_number = 0
    for cnt in larger_contours:
        line_number += 1
        line = e.get_contour_content(cnt, th, False)
        file_name = '{}line_{}.jpg'.format(out_dir_name, line_number)
        line_locations.append(file_name)
        cv2.imwrite(file_name, line)
    return line_locations

def segment_words_from_document(document_location):
    line_locations = segment_lines_from_document(document_location)
    print('All lines from the document has been segmented.')
    for line_location in line_locations:
        segment_words_from_line(line_location)
    print('All words from all of the lines has been segmented.')

PAGE_WIDTH = 600
LINE_HEIGHT = 30

src = 'pages/test.jpg'
if len(sys.argv) == 2:
    src = sys.argv[1]
segment_words_from_document(src)