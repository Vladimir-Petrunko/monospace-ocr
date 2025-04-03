from pathlib import Path

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