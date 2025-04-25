import cv2
import glob
import json
import numpy
import ocr
import time
import editdistance

annotations = glob.glob('D:/codescan/codescan' + '/**/annotations.json', recursive = True)[11:12]

def get_lines_data(annotations_obj, coding_grid_category):
    for annotation in annotations_obj:
        if annotation['category_id'] == int(coding_grid_category):
            return annotation['lines_data']

def analyze_accuracy():
    times = numpy.array([])
    skipped = 0
    total = 0
    for annotation in annotations:
        with open(annotation, 'r') as file:
            data = json.load(file)
            categories = data['categories']
            coding_grid_category = [key for key, value in categories.items() if value == 'coding_grid'][0]
            image_file = 'D:/codescan/codescan/' + data['images'][0].get('file_name') + '/bw.png'
            lines_data = get_lines_data(data['annotations'], coding_grid_category)
            image = cv2.imread(image_file)
            if lines_data is None or image is None:
                skipped = skipped + 1
                continue
            for line in lines_data:
                row_l, col_l = int(line['y']), int(line['x'])
                row_r, col_r = row_l + int(line['height']), col_l + int(line['code_width'])
                start = time.time()
                ground_truth_bounding_box = image[row_l:row_r, col_l:col_r]
                text = ocr.image_to_text(ground_truth_bounding_box)
                end = time.time()
                times = numpy.append(times, end - start)
                print(text, line['text'])
            total = total + 1
    print('Time (s):', 'mean:', numpy.mean(times), 'std:', numpy.std(times))
    print('Skipped images:', skipped, 'out of', total)

analyze_accuracy()