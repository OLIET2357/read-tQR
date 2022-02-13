import cv2
import numpy as np
from qr import QRCodeFinder
import argparse

DEFAULT_TQR_SIZE = 19

parser = argparse.ArgumentParser(description='Read tQR Code', epilog='v1.0')

mutual = parser.add_mutually_exclusive_group(required=True)
mutual.add_argument('-t', '--text-file', help='Text File')
mutual.add_argument('-i', '--image-file', help='Image File')
parser.add_argument('--show-raw-data',
                    help='show raw data', action='store_true')
parser.add_argument('--show-image', help='show image', action='store_true')
parser.add_argument('-s', '--size',
                    help=f'tQR size (default: {DEFAULT_TQR_SIZE})', default=DEFAULT_TQR_SIZE)

args = parser.parse_args()

TEXT_FILE = args.text_file
IMAGE_FILE = args.image_file

SHOW_RAW_DATA = args.show_raw_data
SHOW_IMAGE = args.show_image

TQR_SIZE = args.size


def img2txt(img, SIZE):
    m = [[None for _ in range(SIZE)] for _ in range(SIZE)]

    w, h = img.shape[:2]
    for i in range(SIZE):
        for j in range(SIZE):
            cropped_img = img[round(w * i / SIZE):round(w * (i + 1) / SIZE),
                              round(h * j / SIZE):round(h * (j + 1) / SIZE)]
            mean = cv2.mean(cropped_img)[0]
            m[i][j] = mean < 256 / 2

    return np.rot90(m)


def check_tqr(q):
    from cells import cells

    for i in range(TQR_SIZE):
        for j in range(TQR_SIZE):
            if cells[i][j] is not None:
                if q[i][j] != cells[i][j]:
                    return False

    return True


def decode_tqr(q):
    assert check_tqr(q), 'invalid tqr'

    assert q[1][15] != q[0][14], 'Maybe Left Door'
    x = q[0][14]
    rd1 = ''.join(map(str, map(int, [
        not q[2][18], q[1][17], q[2][17], not q[0][16], q[1]
        [16], q[2][16], q[0][15], x, not q[1][14], not q[2][14]
    ])))

    rd2 = ''.join(map(str, map(int, [
        not q[0][13], q[1][13], q[2][13], not q[0][12], q[1][12],
        q[2][12], not q[2][11], q[0][10], not q[1][10], not q[2][10]
    ])))

    rd3 = ''.join(map(str, map(int, [
        not q[0][9],  q[1][9], q[2][9], not q[0][8], q[3][9],
        not q[4][9], not q[5][9], not q[6][9], q[7][9], q[3][8]
    ])))

    rd4 = ''.join(map(str, map(int, [
        not q[4][8], not q[5][8], not q[3][11], q[4][11], q[5][11],
        q[6][11], not q[7][11], not q[3][10], q[4][10], q[5][10]
    ])))

    if SHOW_RAW_DATA:
        print(rd1, rd2, rd3, rd4)

    d1 = int(rd1, 2)
    d2 = int(rd2, 2)
    d3 = int(rd3, 2)
    d4 = int(rd4, 2)

    return '%03d%03d%03d%03d' % (d1, d2, d3, d4)


if __name__ == '__main__':
    if TEXT_FILE:
        q = []
        for l in open(TEXT_FILE):
            q.append(list(map(lambda c: c == '1', list(l.removesuffix('\n')))))
        print(decode_tqr(q))
    elif IMAGE_FILE:
        finder = QRCodeFinder()
        tqrs = finder.find(IMAGE_FILE)
        tqr = tqrs[0]
        print(decode_tqr(img2txt(tqr.image, TQR_SIZE)))
        if SHOW_IMAGE:
            cv2.imshow('tqr image', tqr.image)
            cv2.waitKey(0)
