import ocr, cv2
import time, numpy, utils, image_codec

ocr.initialize()
start_time = time.time()
print(ocr.encode(cv2.imread('img.png')))
end_time = time.time()
print('Time taken:', end_time - start_time, 's.')

ocr.decode(input_file = 'output.eva')