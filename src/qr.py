# -*- coding: utf-8 -*-
import os
import sys
import traceback
import argparse
import cv2
import numpy as np
import colorsys

# INTER_NEAREST - a nearest-neighbor interpolation
# INTER_LINEAR - a bilinear interpolation (used by default)
# INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
# INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
# INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood


# http://stackoverflow.com/questions/3252194/numpy-and-line-intersections
#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# http://stackoverflow.com/questions/3252194/numpy-and-line-intersections
#
# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# check p is on left side of p0-p1
def is_left_side(p, p0, p1):
    v0 = p1 - p0
    v1 = p - p0
    return  np.cross(v0, v1) < 0

##
class QRCode():
    def __init__(self, points, image):
        self.points = points
        self.image = image

##
class QRCodeFinder():
    def __init__(self):
        self.img = None
        self.threshold = 64
        self.min_width = 640
        self.min_height = 480
        self.output_width= 128
        self.output_height = 128

        self._img_work = None
        self._img_dbg = None

        self._contours = None
        self._hierarchies = None

        self._candidates = None
        self._patterns = None
        self._pattern_center = None

        self._outer_points = None

        self._qr_codes = None

        self._drawer = QRCodeDebugDrawer()
        
        pass

    def find(self, img):
        self._prepare_img(img)
        self._find_contours()
        self._find_pattern_candidates()
        self._find_patterns()
        self._sort_patterns()
        self._get_outer_points()
        return self._build_result()

    def _prepare_img(self, imgname):
        img = cv2.imread(imgname)
        
        w, h = img.shape[:2]

        r = max(self.min_width / w, self.min_height / h)
        if r > 1.0:
            itp = cv2.INTER_LANCZOS4
            img = cv2.resize(img, (int(r * h), int(r * w)), interpolation=itp)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        img_ibin = 255 - img_bin
        img_ibin2 = cv2.morphologyEx(img_ibin, cv2.MORPH_OPEN, kernel)

        self.img = img
        self._img_work = img_ibin2
#        self._img_out = img
        self._img_out = img_bin
        self._img_bin = img_bin
        self._img_dbg = img.copy()

    def _find_contours(self):
        temp = self._img_work.copy()
        contours, hierarchies = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 頂点数を減らす
        new_contours = []
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            new_contours.append(approx)

        self._contours = new_contours
        self._hierarchies = hierarchies

        # cv2.polylines(self._img_dbg, self._contours, True, [0, 0, 255], 1)
        self._drawer.draw_has_child_or_parent(self._img_dbg, self._contours, self._hierarchies)

    def _find_root_node(self, hierarchy, leaf_index, target_depth):
        h = hierarchy[leaf_index]
        root_index = -1
        depth = 1
        # h[3]: parent
        while h[3] != -1 and depth < target_depth:
            root_index = h[3]
            depth += 1
            h = hierarchy[h[3]]
        if h[3] == -1 and depth == target_depth:
            return root_index
        return -1
        
    def _find_pattern_candidates(self):
        contours = self._contours
        hierarchies = self._hierarchies
        candidates = []
        for hierarchy in hierarchies:
            for i, h in enumerate(hierarchy):
                next_cnt, prev_cnt, first_child, parent = h
                if not (first_child == -1 and parent != -1):
                    continue
                root_index = self._find_root_node(hierarchy, i, 3)
                if root_index != -1 and not (root_index in candidates):
                    candidates.append(root_index)

        self._candidates = candidates

        self._drawer.draw_candidates(self._img_dbg, self._contours, self._candidates)

    def _is_valid_pattern(self, pat):
        if len(pat) != 7:
            return False
        a = pat[0]
        if not (pat[2] == pat[3] == pat[4] == pat[6] == a):
            return False
        if not (pat[1] != a and pat[5] != a):
            return False
        return True

    def _contour_to_box(self, cnt):
        box = np.array(list(map(lambda x: x[0], cnt)))
        return box

    def _find_pattern(self, img, cnt):
        box = self._contour_to_box(cnt)
        if len(box) != 4:
            return None
        num = 7
        denom = float(num)
        pts_dbg = []
        # check p0 to p2 and p1 to p3
        for j in range(2):
            x0, y0 = box[j]
            x1, y1 = box[j+2]
            dx, dy = (x1 - x0) / denom, (y1 - y0) / denom
            x0 += dx * 0.5
            y0 += dy * 0.5
            pat = []
            for i in range(num):
                x, y = int(x0 + i * dx), int(y0 + i * dy)
                pat.append(img[y, x])
                pts_dbg.append((x, y))
            if not self._is_valid_pattern(pat):
                box = None
                break

        # debug draw
        for p in pts_dbg:
            cv2.circle(self._img_dbg, p, 1, [255, 255, 0], 2)
            
        return box

    def _find_patterns(self):
        contours = self._contours
        patterns = []
        for c in self._candidates:
            cnt = contours[c]
            box = self._find_pattern(self._img_work, cnt)
            if box is None:
#                print('contour[%d] is not pattern' % c)
                continue
            patterns.append((c, box))

        # TODO: 候補が4個以上の場合の対応
        if len(patterns) > 3:
            patterns = patterns[:3]

        self._patterns = patterns

    # ptterns = [(index0, box0), (index1, box1), (index2, box2)]
    def _find_top_left_pattern(self, patterns):
        max_index = -1
        max_length = 0
        for i, pat in enumerate(patterns):
            _, box0 = patterns[i]
            _, box1 = patterns[(i + 1) % 3]
            c0 = np.mean(box0, axis=0)
            c1 = np.mean(box1, axis=0)
            l = sum((c1 - c0) ** 2)
            if max_index == -1 or l > max_length:
                max_index = (i + 2) % 3
                max_length = l
        return max_index

    def _sort_patterns(self):
        patterns = self._patterns

        if len(patterns) != 3:
            raise Exception('_sort_patterns: [ERROR] len(patterns) != 3')
        
        idx_tl = self._find_top_left_pattern(patterns)
        if idx_tl == -1:
            raise Exception('_sort_patterns [ERROR] cannot find top-left pattern')

        # calculate center point
        idx_tr = (idx_tl + 1) % 3
        idx_bl = (idx_tl + 2) % 3
        c0 = np.mean(patterns[idx_tr][1], axis=0)
        c1 = np.mean(patterns[idx_bl][1], axis=0)
        center = np.mean([c0, c1], axis=0)

        # sort patterns to (top-left, top-right, bottom-left)
        p = np.mean(patterns[idx_tl][1], axis=0)
        if not is_left_side(c0, p, center):
            idx_tr, idx_bl = idx_bl, idx_tr

        indices = [idx_tl, idx_tr, idx_bl]

        self._patterns = [patterns[i] for i in indices] # sort
        self._pattern_center = center

        # draw top-left pattern with blue color
        cv2.polylines(self._img_dbg, [self._patterns[0][1]], True, [255, 0, 0], 2)
        # draw center with green color
        temp = np.int32(self._pattern_center)
        cv2.circle(self._img_dbg, (temp[0], temp[1]), 4, [0, 255, 0], 2)
        
    def _get_outer_point(self, points, center):
        max_index = -1
        max_length = 0
        for i, p in enumerate(points):
            l = sum((center - p) ** 2)
            if max_index == -1 or l > max_length:
                max_index = i
                max_length = l
        return max_index

    def _get_outer_points(self):
        patterns = self._patterns
        center = self._pattern_center
        outer_points = []
        
        # top-left
        points = patterns[0][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])

        # top-right
        points = patterns[1][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])
        idx_next = (idx + 1) % 4
        if not is_left_side(points[idx_next], points[idx], points[(idx + 2) % 4]):
            idx_next = (idx + 3) % 4
        p_tr = points[idx_next]

        # bottom-left
        points = patterns[2][1]
        idx = self._get_outer_point(points, center)
        outer_points.append(points[idx])
        idx_next = (idx + 1) % 4
        if is_left_side(points[idx_next], points[idx], points[(idx + 2) % 4]):
            idx_next = (idx + 3) % 4
        p_bl = points[idx_next]

        # calculate bottom-right point
        try:
            p_br = seg_intersect(outer_points[1], p_tr, outer_points[2], p_bl)
            p_br = np.int32(p_br)
            outer_points.append(p_br)
        except:
            pass

        self._outer_points = outer_points

        # draw outer points
        for p in outer_points:
            cv2.circle(self._img_dbg, (p[0], p[1]), 4, [0, 255, 0], 2)
        cv2.circle(self._img_dbg, (p_tr[0], p_tr[1]), 4, [0, 255, 255], 2)
        cv2.circle(self._img_dbg, (p_bl[0], p_bl[1]), 4, [0, 255, 255], 2)

    def _crop_qr_code(self, img_src, pts_src, width, height):
        num = min(len(pts_src), 4)
        pts_src = np.array(pts_src[:num], dtype=np.float32)
        pts_dst = [(0, 0), (width, 0), (0, height)]
        if num == 3:
            pts_dst = np.array(pts_dst, dtype=np.float32).reshape(3, 2)
            M = cv2.getAffineTransform(pts_src, pts_dst)
            img_dst = cv2.warpAffine(img_src, M, (height, width))
            return img_dst
        if num == 4:
            pts_dst.append((width, height))
            pts_dst = np.array(pts_dst, dtype=np.float32).reshape(4, 2)
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            img_dst = cv2.warpPerspective(img_src, M, (height, width))
            return img_dst
        return None

    def _build_result(self):
        qrs = []

        w, h = self.output_width, self.output_height
        img = self._crop_qr_code(self._img_out, self._outer_points, w, h)

        if not (img is None):
            # TODO: 最小サイズの考慮
            qr = QRCode(self._outer_points, img)
            qrs.append(qr)
        
        return qrs
        
    ##
    def show_bin(self, title='bin', wait=True):
        cv2.imshow(title, self._img_bin)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAll

    def show_debug(self, title='img', wait=True):
        cv2.imshow(title, self._img_dbg)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

##
class QRCodeDebugDrawer():
    def __init__(self):
        pass

    def draw_polyline(self, img, cnt, isClosed=True, color=[0, 0, 0], thickness=1):
        length = len(cnt)
        if length < 2:
            return
        p0 = tuple(cnt[0][0][:2])
        first = p0
        for i in range(1, length):
            p1 = tuple(cnt[i][0][:2])
            cv2.line(img, p0, p1, color, thickness)
            p0 = p1
        if isClosed:
            cv2.line(img, p0, first, color, thickness)

    # draw only has child or parent contours
    def draw_has_child_or_parent(self, img, contours, hierarchies):
        for hierarchy in hierarchies:
            for i, h in enumerate(hierarchy):
                next_cnt, prev_cnt, first_child, parent = h
                if first_child == -1 and parent == -1:
                    continue
                cnt = contours[i]
                # cv2.polylines(img, cnt, True, [0, 0, 255], 1)
                self.draw_polyline(img, cnt, True, [0, 0, 255], 1)

    def draw_candidates(self, img, contours, candidates):
        for c in candidates:
            cnt = contours[c]
            self.draw_polyline(img, cnt, True, [0, 255, 255], 1)

        
def parse_args(args):
    parser = argparse.ArgumentParser('Find QR code')

    parser.add_argument('image',
                        metavar='IMAGE',
                        type=str,
                        default=None,
                        help='input image')

    default_threshold = 127
    parser.add_argument('-t', '--threshold',
                        metavar='VALUE',
                        type=int,
                        dest='threshold',
                        default=default_threshold,
                        help='threshold value for binarization (default: %d)' % default_threshold)

    parser.add_argument('-o', '--output',
                        metavar='FILE',
                        type=str,
                        dest='output',
                        default=None,
                        help='output file path')

    default_output_size = 128
    parser.add_argument('-s', '--output-size',
                        metavar='SIZE',
                        type=int,
                        dest='output_size',
                        default=default_output_size,
                        help='output file size (default: %d)' % default_output_size)

    default_debug_log = False
    parser.add_argument('-dl', '--debug-log',
                        metavar='VALUE',
                        dest='debug_log',
                        action='store_const',
                        const=True,
                        default=default_debug_log,
                        help='output debug log or not (default: %d)' % default_debug_log)

    default_debug_image = False
    parser.add_argument('-di', '--debug-image',
                        metavar='VALUE',
                        dest='debug_image',
                        action='store_const',
                        const=True,
                        default=default_debug_image,
                        help='show debug image or not (default: %d)' % default_debug_image)

    return parser.parse_args(args)


if __name__ == '__main__':
    args  = parse_args(sys.argv[1:])
    print(args)
    
    finder = QRCodeFinder()
    finder.threshold = args.threshold
    finder.output_width = args.output_size
    finder.output_height = args.output_size

    qrs = []
    try:
        qrs = finder.find(args.image)
    except:
        if args.debug_log:
            traceback.print_exc()

    print('%d QR code(s) found' % len(qrs))
    for qr in qrs:
        # dump region of QR code
        print(', '.join(['(%d, %d)' % (p[0], p[1]) for p in qr.points]))

    # TODO: 複数画像の出力
    if (not (args.output is None)) and len(qrs) > 0:
        qr = qrs[0]
        path = args.output
        if os.path.exists(path):
            if os.path.isdir(path):
                print('[ERROR] %s is directory' % path)
                path = None
            else:
                if raw_input('override file %s ? (y/N)' % path) != 'y':
                    path = None
        if not (path is None):
            cv2.imwrite(path, qr.image)

    if args.debug_image:
        finder.show_bin(wait=False)
        finder.show_debug()
                
#     for i, qr in enumerate(qrs):
#         cv2.imshow('qr_%d' % i, qr.image)
#  
#     print('Press some key to exit')
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    pass