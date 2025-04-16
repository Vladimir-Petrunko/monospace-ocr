import math
import dahuffman
import numpy
from dahuffman import HuffmanCodec

frequencies = {chr(c): 1 for c in range(33, 127)}

class Region:
    def __init__(self, split_size, splits, symbols, styles, foregrounds, backgrounds):
        self.split_size = split_size
        self.splits = splits
        self.symbols = symbols
        self.styles = styles
        self.foregrounds = foregrounds
        self.backgrounds = backgrounds

def arr_8_to_int(arr):
    return sum(int(x) << i for i, x in enumerate(arr))

def arr_to_ints(arr):
    result = []
    for i in range(0, len(arr), 8):
        result.append(arr_8_to_int(arr[i:(i + 8)]))
    return result

def encode(regions):
    huffman_codec = HuffmanCodec.from_frequencies(frequencies)

    data = numpy.array([], dtype = 'uint8')
    for region in regions:
        for axis in range(2):
            # Split sizes
            data = numpy.append(data, list(region.split_size[axis].to_bytes(2)))
            data = numpy.append(data, list(region.splits[axis][0].to_bytes(2)))
            # Split coordinates (delta compared to the expected step size)
            splits = numpy.array([])
            for i in range(1, len(region.splits[axis])):
                delta = (region.splits[axis][i] - region.splits[axis][i - 1]) - region.split_size[axis]
                if delta == -1:
                    splits = numpy.append(splits, [1, 0])
                elif delta == 0:
                    splits = numpy.append(splits, [0])
                else:
                    splits = numpy.append(splits, [1, 1])
            splits_bytes = arr_to_ints(splits)
            data = numpy.append(data, list(len(splits_bytes).to_bytes(2)))
            data = numpy.append(data, splits_bytes)
        # Symbols
        symbols = ''.join(region.symbols)
        symbols_bytes = list(huffman_codec.encode(symbols))
        data = numpy.append(data, list(len(symbols_bytes).to_bytes(4)))
        data = numpy.append(data, symbols_bytes)
        # Styles
        styles = numpy.array([])
        for style in region.styles:
            if style == 'r':
                styles = numpy.append(styles, [0])
            elif style == 'b':
                styles = numpy.append(styles, [1, 0])
            else:
                styles = numpy.append(styles, [1, 1])
        styles_bytes = arr_to_ints(styles)
        data = numpy.append(data, list(len(styles_bytes).to_bytes(4)))
        data = numpy.append(data, styles_bytes)
        # Colors
        color_to_index = {}
        color_bytes = numpy.array([], dtype = 'uint8')
        byte_sz = 1
        for color in (numpy.concat((region.backgrounds, region.foregrounds))):
            color = tuple(color)
            if color in color_to_index:
                color_bytes = numpy.append(color_bytes, [0])
            else:
                if len(color_to_index) == 256:
                    color_bytes = numpy.append(color_bytes, [1, 0])
                    byte_sz = byte_sz + 1
                else:
                    color_bytes = numpy.append(color_bytes, [1, 1])
                color_to_index[color] = len(color_to_index)
                color_bytes = numpy.append(color_bytes, [int(color[0]), int(color[1]), int(color[2])])
            color_bytes = numpy.append(color_bytes, list(color_to_index[color].to_bytes(byte_sz)))
        data = numpy.append(data, list(len(color_bytes).to_bytes(4)))
        data = numpy.append(data, color_bytes)

    return bytes(list(data))