from pathlib import Path
import cv2

def calculate_area(pixels, rows, columns, color):
    area = 0

    for row in rows:
        for col in columns:
            if pixels[row][col] == color:
                area = area + 1

    return area

def to_black_and_white(image):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    return image

def get_background(image):
    height = image.shape[0]
    width = image.shape[1]
    white_area = calculate_area(image, range(0, height), range(0, width), 255)
    if white_area * 2 >= height * width:
        return 255
    return 0

def get_bounding_box(image, background):
    height = image.shape[0]
    width = image.shape[1]
    row_l = 0
    row_r = height
    col_l = 0
    col_r = width

    while col_l < col_r and row_l < row_r:
        end_shrink = True

        if calculate_area(image, range(row_l, row_l + 1), range(col_l, col_r), background) == col_r - col_l:
            row_l = row_l + 1
            end_shrink = False
        if calculate_area(image, range(row_r - 1, row_r), range(col_l, col_r), background) == col_r - col_l:
            row_r = row_r - 1
            end_shrink = False
        if calculate_area(image, range(row_l, row_r), range(col_l, col_l + 1), background) == row_r - row_l:
            col_l = col_l + 1
            end_shrink = False
        if calculate_area(image, range(row_l, row_r), range(col_r - 1, col_r), background) == row_r - row_l:
            col_r = col_r - 1
            end_shrink = False

        if end_shrink:
            break

    return row_l, row_r, col_l, col_r

def file_cnt(path):
    path = Path(path)
    return sum(1 for entry in path.iterdir() if entry.is_file())

def draw_vertical_line(image, column, color):
    height = image.shape[0]
    for i in range(height):
        image[i][column] = color
    return image

def draw_horizontal_line(image, row, color):
    width = image.shape[1]
    for j in range(width):
        image[row][j] = color
    return image