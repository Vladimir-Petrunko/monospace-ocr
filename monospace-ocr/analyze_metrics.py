# This file provides a mechanism to analyze various metrics of the OCR compressor:
# 1. Speed (total speed & speed without compressing)
# 2. Accuracy (i.e. CER)
# 3. Compression size (%)

import cv2
import editdistance
import time
import ocr_client
import os

def calculate_metrics(file_name, true_text):
    """
    Calculates the OCR compressor metrics
    :param file_name: the filename of the image to be processed.
    :param true_text: the true text on the image
    """
    start = time.time()
    image = cv2.imread(file_name)
    parsed = ocr_client.parse(image)
    # After OCR parsing
    middle = time.time()
    ocr_client.encode_aux(image, parsed, 'Consolas', 'auxiliary.eva')
    end = time.time()
    print('Time (s): total', end - start, ', only OCR', middle - start)

    # Calculate CER
    errors = 0
    totals = 0
    parsed_text = ocr_client.image_to_text(file_name).split('\n')
    true_text = true_text.split('\n')
    if len(parsed_text) != len(true_text):
        raise Exception('Parsed and true text have different line counts!')
    length = len(true_text)
    for i in range(length):
        errors = errors + editdistance.eval(true_text[i], parsed_text[i])
        totals = totals + len(true_text[i])
    print('CER:', errors / totals)

    # Calculate compression rate
    original_bytes = image.nbytes
    compressed_bytes = os.path.getsize('auxiliary.eva')
    print('Compression rate:', 100 - (compressed_bytes / original_bytes * 100))