import cv2
from qr import QRCodeFinder
import sys

SHOW_IMAGE = False


def img2txt(img, SIZE):
    m = [[None for _ in range(SIZE)] for _ in range(SIZE)]

    w, h = img.shape[:2]
    for i in range(SIZE):
        for j in range(SIZE):
            cropped_img = img[round(w * i / SIZE):round(w *
                                                        (i + 1) / SIZE), round(h * j / SIZE):round(h * (j + 1) / SIZE)]
            mean = cv2.mean(cropped_img)[0]
            m[i][j] = int(mean < 256 / 2)
    return m


def m(x):
    return str(1 - int(x, 2))


def check_tqr(q):
    return True


def decode_tqr(q):
    sq = []
    for l in q:
        sq.append(list(map(str, l)))
    d1 = int(m(sq[18][16]) + sq[17][17] + sq[17][16] + m(sq[16][18]) + sq[16][17] +
             sq[16][16] + sq[15][18] + sq[14][18] + m(sq[14][17]) + m(sq[14][16]), 2)
    d2 = int(m(sq[13][18]) + sq[13][17] + sq[13][16] + m(sq[12][18]) + sq[12][17] +
             sq[12][16] + m(sq[11][16]) + sq[10][18] + m(sq[10][17]) + m(sq[10][16]), 2)
    d3 = int(m(sq[9][18]) + sq[9][17] + sq[9][16] + m(sq[8][18]) + sq[9][15] +
             m(sq[9][14]) + m(sq[9][13]) + m(sq[9][12]) + sq[9][11] + sq[8][15], 2)
    d4 = int(m(sq[8][14]) + m(sq[8][13]) + m(sq[11][15]) + sq[11][14] + sq[11][13] +
             sq[11][12] + m(sq[11][11]) + m(sq[10][15]) + sq[10][14] + sq[10][13], 2)

    return '%03d%03d%03d%03d' % (d1, d2, d3, d4)


if len(sys.argv) == 1:
    print(sys.argv[0], '[IMG]')
    exit(1)

finder = QRCodeFinder()

qrs = finder.find(sys.argv[1])

for qr in qrs:
    print(decode_tqr(img2txt(qr.image, 19)))
    if SHOW_IMAGE:
        cv2.imshow('', qr.image)
        cv2.waitKey(0)
