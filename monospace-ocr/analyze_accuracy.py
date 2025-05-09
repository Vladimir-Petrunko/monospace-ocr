import cv2
import glob
import json
import numpy
import ocr
import time
import editdistance
import pytesseract
import easyocr

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
annotations = glob.glob('D:/codescan/codescan' + '/**/annotations.json', recursive = True)[:1]

def get_lines_data(annotations_obj, coding_grid_category):
    for annotation in annotations_obj:
        if annotation['category_id'] == int(coding_grid_category):
            return annotation['lines_data']

def analyze_accuracy():
    times = numpy.array([])
    errors = numpy.array([])
    passed = 0
    for annotation in annotations:
        if passed % 10 == 0:
            print('Passed:', passed)
        with open(annotation, 'r') as file:
            data = json.load(file)
            categories = data['categories']
            coding_grid_category = [key for key, value in categories.items() if value == 'coding_grid'][0]
            image_file = 'D:/codescan/codescan/' + data['images'][0].get('file_name') + '/bw.png'
            lines_data = get_lines_data(data['annotations'], coding_grid_category)
            image = cv2.imread(image_file)
            if lines_data is None or image is None:
                continue
            for line in lines_data:
                row_l, col_l = int(line['y']), int(line['x'])
                row_r, col_r = row_l + int(line['height']), col_l + int(line['code_width'])
                start = time.time()
                if row_l < 0 or row_r >= image.shape[0] or col_l < 0 or col_r >= image.shape[1]:
                    continue
                ground_truth_bounding_box = image[row_l:row_r, col_l:col_r]
                tesseract_text = pytesseract.image_to_string(ground_truth_bounding_box)
                tesseract_text = tesseract_text.replace('\n', '')
                while tesseract_text.startswith(' '):
                    tesseract_text = tesseract_text[1:]
                ground_truth_text = line['text']
                ground_truth_text = ground_truth_text.replace('\n', '')
                while '  ' in ground_truth_text:
                    ground_truth_text = ground_truth_text.replace('  ', ' ')
                while ground_truth_text.startswith(' '):
                    ground_truth_text = ground_truth_text[1:]
                # text = ocr.image_to_text(ground_truth_bounding_box)[3]
                print(tesseract_text)
                print(ground_truth_text)
                print()
                end = time.time()
                times = numpy.append(times, end - start)
                errors = numpy.append(errors, 100.0 * max(0.0, 1.0 - (editdistance.eval(tesseract_text, ground_truth_text)) / len(ground_truth_text)))
        passed = passed + 1
    errors = numpy.sort(errors)
    errors = errors[4:]
    print('Time (s):', 'mean:', numpy.mean(times), 'std:', numpy.std(times))
    print('CER:', 'mean:', numpy.mean(errors), 'std:', numpy.std(errors), 'p90:', numpy.percentile(errors, 10), 'p99:', numpy.percentile(errors, 1))
    print(numpy.sort(errors))
analyze_accuracy()