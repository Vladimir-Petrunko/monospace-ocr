import numpy

import utils

# This algorithm takes an image with monospace text and creates a grid that aligns itself optimally
# between symbol borders. It is rather insensitive to garbage data outside the monospace area, as
# long as there isn't _too_ much of it. It is also rather insensitive to the two hyperparameters
# given below, so it's possible to tweak them within reasonable margins.

# Accepted symbol widths.
COL_SPLIT_RANGE = range(5, 25)
# Accepted difference (along all color channels) between 2 pixels for them to be considered 'similar'
SIMILAR_THRESHOLD = 30

def similar(a, b):
    cnt = len(a)
    for i in range(cnt):
        if abs(a[i] - b[i]) > SIMILAR_THRESHOLD:
            return False
    return True

# A _run_ is defined as a change in color between 2 consecutive pixels.
# Note that two colors within SIMILAR_THRESHOLD of each other are considered identical.
def find_runs(image):
    horizontal_runs = numpy.zeros(image.shape)
    vertical_runs = numpy.zeros(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    for i in range(1, height):
        for j in range(1, width):
            if not similar(image[i][j], image[i - 1][j]):
                vertical_runs[i][j] = 1
            if not similar(image[i][j], image[i][j - 1]):
                horizontal_runs[i][j] = 1
    return horizontal_runs, vertical_runs

def find_optimal_split(runs_cnt, split_range):
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
            if prev_result * 1.25 < best_split_result:
                prev_prev_result = results_by_index[split - 2]
                if prev_prev_result < prev_result:
                    return splits_by_index[split - 2]
                return splits_by_index[split - 1]

        results_by_index[split] = best_split_result
        splits_by_index[split] = best_splits

    return None

def draw_splits(image, vertical_splits, horizontal_splits, color):
    for split in vertical_splits:
        image = utils.draw_vertical_line(image, split, color)
    for split in horizontal_splits:
        image = utils.draw_horizontal_line(image, split, color)
    return image

def create_grid(image):
    image = image.astype(numpy.int16)
    height = image.shape[0]
    width = image.shape[1]

    horizontal_runs, vertical_runs = find_runs(image)
    vertical_runs_cnt = [numpy.sum(vertical_runs[:, i:(i + 1)]) for i in range(width)]
    horizontal_runs_cnt = [numpy.sum(horizontal_runs[i:(i + 1), :]) for i in range(height)]

    vertical_splits = find_optimal_split(vertical_runs_cnt, COL_SPLIT_RANGE)
    vertical_split_width = vertical_splits[0] - vertical_splits[1]
    horizontal_split_range = range(int(vertical_split_width * 1), int(vertical_split_width * 4))
    horizontal_splits = find_optimal_split(horizontal_runs_cnt, horizontal_split_range)

    image = draw_splits(image, vertical_splits, horizontal_splits, (0, 0, 255))
    return image
