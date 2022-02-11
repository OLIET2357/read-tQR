import sys
from glob import glob

DATA = True
GARBAGE = False


def m(x):
    return str(1 - int(x, 2))


def main(filename):
    print('filename', filename)
    q = open(filename).read().split()
    if DATA:
        x = m(q[1][15])
        x = q[0][14]
        assert m(q[1][15]) == q[0][14]
        d1 = int(m(q[2][18]) + q[1][17] + q[2][17] + m(q[0][16]) + q[1][16] +
                 q[2][16] + q[0][15] + x + m(q[1][14]) + m(q[2][14]), 2)
        d2 = int(m(q[0][13]) + q[1][13] + q[2][13] + m(q[0][12]) + q[1][12] +
                 q[2][12] + m(q[2][11]) + q[0][10] + m(q[1][10]) + m(q[2][10]), 2)
        d3 = int(m(q[0][9]) + q[1][9] + q[2][8] + q[2][9] + q[3][9] +
                 m(q[4][9]) + m(q[5][9]) + m(q[6][9]) + q[7][9] + q[3][8], 2)
        d4 = int(m(q[4][8]) + m(q[5][8]) + m(q[3][11]) + q[4][11] + q[5][11] +
                 q[6][11] + m(q[7][11]) + m(q[3][10]) + q[4][10] + q[5][10], 2)
        print('data %03d%03d%03d%03d' % (d1, d2, d3, d4))

    if GARBAGE:
        gl = q[0][18] + m(q[1][18]) + m(q[0][17])
        ga = m(q[1][15]) + m(q[2][15])
        gb = q[0][11] + m(q[1][11])
        gc = m(q[0][8]) + q[1][8]
        gd = m(q[6][8]) + q[7][8]
        ge = q[6][10] + q[7][10]
        print('garbages', gl, ga, gb, gc, gd, ge)

    print()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print(sys.argv[0], '[tQR text] or "all"')
        exit(-1)
    if sys.argv[1] == 'all':
        for filename in glob('tqr_txt\\*.txt'):
            main(filename)
        exit()
    main(sys.argv[1])
