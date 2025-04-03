import cv2

import text_grid

img = cv2.imread('inp.png')
img = text_grid.create_grid(img)
cv2.imwrite('res.png', img)