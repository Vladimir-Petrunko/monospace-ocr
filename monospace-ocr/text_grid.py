import cv2
import numpy, utils
from constants import CHAR_HEIGHT

# This algorithm takes an image with monospace text and creates a grid that aligns itself optimally
# between symbol borders. It is rather insensitive to garbage data outside the monospace area, as
# long as there isn't _too_ much of it. It is also rather insensitive to the two hyperparameters
# given below, so it's possible to tweak them within reasonable margins.

# Accepted symbol widths.
COL_SPLIT_RANGE = range(4, 20)
# Accepted difference (along all color channels) between 2 pixels for them to be considered 'similar'
SIMILAR_THRESHOLD = 40

def is_run(row):
    cnt = len(row) // 2
    for i in range(cnt):
        if abs(row[i] - row[i + cnt]) > SIMILAR_THRESHOLD:
            return 1
    return 0

# A _run_ is defined as a change in color between 2 consecutive pixels.
# Note that two colors within SIMILAR_THRESHOLD of each other are considered identical.
def find_runs(image):
    diff_horizontal = numpy.abs(image[:, 1:] - image[:, :-1]).max(axis = 2) > SIMILAR_THRESHOLD
    diff_vertical = numpy.abs(image[1:] - image[:-1]).max(axis = 2) > SIMILAR_THRESHOLD

    horizontal_runs = numpy.zeros(image.shape[:2], dtype = int)
    vertical_runs = numpy.zeros(image.shape[:2], dtype = int)

    horizontal_runs[:, 1:][diff_horizontal] = 1
    vertical_runs[1:][diff_vertical] = 1

    return horizontal_runs, vertical_runs

def find_optimal_split(runs_cnt, split_range, split_type):
    sz = len(runs_cnt)
    results_by_index = {}
    splits_by_index = {}
    for split in split_range:
        best_split_result = None
        best_splits = None
        for start in range(split):
            # Given: split width [split] and first index [start], find the optimal placement of grid lines
            # with distance split +/- 1 of each other, to minimize the total number of _runs_ (see definition
            # above) across all grid lines.

            # dp_result[i] = best result (sum of total runs) if the last grid line is placed on the i-th index
            dp_result = [-1 for _ in range(sz)]
            # dp_prev_index[i] = the previous grid line in such an optimal configuration (see definition of dp_result[i])
            dp_prev_index = [-1 for _ in range(sz)]
            for i in range(split):
                dp_result[i] = runs_cnt[i]
            for i in range(split, sz):
                for delta in [-1, 0, 1]:
                    prev = i - split + delta
                    if prev < 0:
                        continue
                    if dp_result[i] == -1 or dp_result[prev] + runs_cnt[i] < dp_result[i]:
                        dp_result[i] = dp_result[prev] + runs_cnt[i]
                        dp_prev_index[i] = prev
            best_result = dp_result[sz - 1]
            best_index = sz - 1
            for i in range(split):
                if dp_result[sz - i - 1] < best_result:
                    best_result = dp_result[sz - i - 1]
                    best_index = sz - i - 1
            splits = []
            while best_index != -1:
                splits.append(best_index)
                best_index = dp_prev_index[best_index]
            if best_split_result is None or best_result < best_split_result:
                best_split_result = best_result
                best_splits = splits
        if best_split_result < 1:
            return split, best_splits
        # Generally, if the split width increases, the total run sum decreases (because fewer grid lines can be placed).
        # If it increases and this increase is statistically significant (here > 1.25x), this means that the previous
        # width allowed for a configuration with significantly fewer runs. We declare the first such position our
        # target optimal width.
        # I cannot formally prove this works, but this seems rather obvious.
        # Increase the multiplier with caution, since this implies a stricter condition for our target width. This may
        # lead to undetermined or wrongly determined grids.
        # I believe this multiplier can be decreased +/- safely up to a point, but have not tried it yet.
        if (split - 1) in results_by_index:
            prev_result = results_by_index[split - 1]
            coefficient = 1.1 if split_type == 'vertical' else 3
            if prev_result * coefficient < best_split_result:
                prev_prev_result = results_by_index.get(split - 2, None)
                if prev_prev_result is not None and prev_prev_result < prev_result:
                    return split - 2, splits_by_index[split - 2]
                return split - 1, splits_by_index[split - 1]

        results_by_index[split] = best_split_result
        splits_by_index[split] = best_splits

        if splits_by_index[split] == 0:
            return split, results_by_index[split]

    return None

def draw_splits(image, vertical_splits, horizontal_splits, color):
    for split in vertical_splits:
        for i in range(image.shape[0]):
            image[i][split] = color
    for split in horizontal_splits:
        for j in range(image.shape[1]):
            image[split][j] = color
    return image

def parse_cells(image):
    image = image.astype('int')
    height = image.shape[0]
    width = image.shape[1]

    horizontal_runs, vertical_runs = find_runs(image)
    vertical_runs_cnt = [numpy.sum(vertical_runs[:, i:(i + 1)]) for i in range(width)]
    horizontal_runs_cnt = [numpy.sum(horizontal_runs[i:(i + 1), :]) for i in range(height)]

    ver_w, vertical_splits = find_optimal_split(vertical_runs_cnt, COL_SPLIT_RANGE, 'vertical')
    vertical_splits = numpy.append(vertical_splits, 0)
    vertical_split_width = vertical_splits[0] - vertical_splits[1]
    horizontal_split_range = range(int(vertical_split_width * 1.8), int(vertical_split_width * 3))
    hor_w, horizontal_splits = find_optimal_split(horizontal_runs_cnt, horizontal_split_range, 'horizontal')
    horizontal_splits = numpy.append(horizontal_splits, 0)

    cells = []
    coordinates = []

    horizontal_splits = numpy.flip(horizontal_splits)
    vertical_splits = numpy.flip(vertical_splits)

    for i in range(0, len(horizontal_splits)):
        for j in range(0, len(vertical_splits)):
            row_l, row_r = vertical_splits[j - 1], vertical_splits[j]
            col_l, col_r = horizontal_splits[i - 1] + 1, horizontal_splits[i] - 1
            symbol = image[col_l:(col_r + 1), row_l:(row_r + 1)]
            cells.append(symbol)
            coordinates.append((row_l, row_r, col_l, col_r))

    return ver_w, hor_w, vertical_splits, horizontal_splits, cells
