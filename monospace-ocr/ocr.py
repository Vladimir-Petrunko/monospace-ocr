import ocr_client
import sys

DEFAULT_FONT = 'Consolas'

def parse_mode(args):
    if 'mode=ocr' in args:
        return 'ocr'
    elif 'mode=encode' in args:
        return 'encode'
    elif 'mode=decode' in args:
        return 'decode'
    else:
        raise Exception('Error: expected parameter: mode; should be one of {ocr, encode, decode}.')

def parse_param(param_name, args, required, default = None):
    for arg in args:
        if param_name + '=' in arg:
            return arg[(len(param_name) + 1):]
    if not required:
        return default
    raise Exception('Error: expected parameter: ', param_name, '.')

mode = parse_mode(sys.argv)
if mode == 'ocr':
    input_file = parse_param('input', sys.argv, required = True)
    output_file = parse_param('output', sys.argv, required = False)
    ocr_result = ocr_client.image_to_text(input_file)
    if output_file is None:
        print(ocr_result)
    else:
        with open(output_file, 'w') as file:
            file.write(ocr_result)
elif mode == 'encode':
    input_file = parse_param('input', sys.argv, required = True)
    output_file = parse_param('output', sys.argv, required = True)
    font = parse_param('font', sys.argv, required = False, default = DEFAULT_FONT)
    detail = parse_param('detail', sys.argv, required = False, default = 'none')
    ocr_client.encode(input_file, font, output_file)
else: # mode == 'decode'
    input_file = parse_param('input', sys.argv, required = True)
    output_file = parse_param('output', sys.argv, required = True)
    ocr_client.decode(input_file, output_file)