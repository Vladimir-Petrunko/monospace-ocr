# Monospace OCR | image codec
This project is a dual-function new state-of-the-art algorithm for monospace code recognition (as well as generic monospace text), as well as a hyper-compressing lossless codec for monospace text images (up to 2% if only text data is to be encoded, otherwise 20-80% depending on text area on image).

Created by Vladimir Petrunko @ ITMO University, vladimirr.sapfire13@gmail.com.

## Requirements

Requirements are detailed in `requirements.txt`.

## Usage

### Text recognition on image
`ocr.py <input> mode=recognize output-file=<output>`, where:
* `<input>` is the relative path of the input image, in standard formats like `.jpg`, `.png` or others,
* `<output>` is the relative path of the file where the answer is to be written. If the file exists, it will be overwritten with the parsed text, otherwise it will be created. _If the parameter is not specified, the parsed text will be written to standard output._

### Image encoding
`ocr.py <input> mode=encode detail={all|none} output-file=<output>`, where:
* `<input>` is the relative path of the input image, in standard formats like `.jpg`, `.png` or others,
* `<output>` is the relative path of the file where the encoded image is to be written. If the file exists, it will be overwritten with the encoded image, otherwise it will be created. _If the parameter is not specified, a file with the name `output.eva` will be created in the working directory._
* `<detail>` is one of:
  * `all` -- the entire image will be encoded. All data that is not part of the recognized text block will be encoded "as-is", which will result in much lower compression rates, especially if text concentration is low.
  * `text` -- only the text area will be encoded. This results in hyper-compression rates up to 2% and lower in best-case scenarios, and around 5% in worst-case scenarios.
  The default value is `text`.

### Image decoding
`ocr.py <input> mode=decode output-file=<output>`, where:
* `<input>` is the relative path of the _encoded data_,
* `<output>` is the relative path of the image which will contain the decoded data. If the file exists, it will be overwritten with the decoded image, otherwise it will be created. _If the parameter is not specified, a file with the name `output.jpg` will be created in the working directory._

These methods may also be called directly from Python code. All usage examples are available in `main.py`.

## Examples
Several playground examples are available in the `examples/` directory.