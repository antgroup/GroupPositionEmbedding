# -*- encoding: utf-8 -*-
import os
import json
import numpy as np
import base64
import io
import math
import argparse
from PIL import Image
import shutil
import cv2
from shapely.geometry import Polygon as plg

np.random.seed(0)


class BboxProcessor:
    def __init__(self):
        pass

    def box_norm(self, box, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        box = np.array(box).flatten()
        if len(box) == 8:
            x0, y0, x1, y1 = min(box[::2]), min(box[1::2]), max(box[::2]), max(box[1::2])
        else:
            x0, y0, x1, y1 = box
        x0 = clip(0, int((x0 / (width + 1e-10)) * 1000), 1000)
        y0 = clip(0, int((y0 / (height + 1e-10)) * 1000), 1000)
        x1 = clip(0, int((x1 / (width + 1e-10)) * 1000), 1000)
        y1 = clip(0, int((y1 / (height + 1e-10)) * 1000), 1000)
        assert x1 >= x0
        assert y1 >= y0
        return [x0, y0, x1, y1]

    def get_position_ids(self, segment_ids):
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids

    def random_split_merge(self, texts, bboxes, char_bboxes, max_split_num=3, max_merge_num=3, remove_blank=False):
        def find_same_line(anchor, anchor_text, bboxes):
            anchor, bboxes = np.array(anchor), np.array(bboxes)
            char_width = (abs(anchor[1,0] - anchor[0,0]) + abs(anchor[2,0] - anchor[3,0])) / (2 * len(anchor_text))
            y_upper, y_bottom = np.min(anchor[:, 1]), np.max(anchor[:, 1])
            expand = int((y_bottom - y_upper) * 0.5)
            y_upper -= expand
            y_bottom += expand
            y_mins, y_maxs = np.min(bboxes[:, :, 1], axis=1), np.max(bboxes[:, :, 1], axis=1)
            bboxes_ids = set(np.where(y_mins >= y_upper)[0]).intersection(set(np.where(y_maxs <= y_bottom)[0]))
            bboxes_ids = sorted(bboxes_ids, key=lambda x: bboxes[x][0][0]) # sort by x
            return bboxes_ids

        def merge_bboxes(merge_id, texts, bboxes, used, max_merge_part=3):
            if used[merge_id]:
                return texts, bboxes, []
            anchor = bboxes[merge_id]
            anchor_text = texts[merge_id]
            same_line_bboxes_ids = find_same_line(anchor, anchor_text, bboxes)
            same_line_bboxes_ids = list(filter(lambda x: used[x] == 0, same_line_bboxes_ids)) # ensure one box only merge once
            if len(same_line_bboxes_ids) <= 1:
                return texts, bboxes, []

            merge_part_num = len(same_line_bboxes_ids) if len(same_line_bboxes_ids) == 2 else np.random.randint(2, max_merge_part+1)
            total_num = len(same_line_bboxes_ids)
            if merge_part_num < total_num:
                cur_idx = same_line_bboxes_ids.index(merge_id)
                left_num = np.random.randint(min(cur_idx, merge_part_num)) if cur_idx > 0 else 0
                same_line_bboxes_ids = same_line_bboxes_ids[cur_idx-left_num: cur_idx-left_num+merge_part_num]
            merge_text, merge_bbox = '', []
            for box_id in same_line_bboxes_ids:
                used[box_id] = 1
                merge_text += texts[box_id]
                merge_bbox.append(bboxes[box_id])
            
            merge_bbox = np.array(merge_bbox)
            x_min, y_min, x_max, y_max = np.min(merge_bbox[:, :, 0]), np.min(merge_bbox[:, :, 1]), np.max(merge_bbox[:, :, 0]), np.max(merge_bbox[:, :, 1])
            merge_bbox = [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
            texts[merge_id], bboxes[merge_id] = merge_text, merge_bbox
            same_line_bboxes_ids.remove(merge_id)
            return texts, bboxes, same_line_bboxes_ids

        def split_bboxes(text, bbox, char_bbox, max_split_part=3):
            
            split_part_num = np.random.randint(1, max(2, min(int(len(text) // 2), max_split_part)))
            split_pos = sorted(np.random.choice(range(1, len(text)), split_part_num, replace=False)) + [len(text)]
            split_text, split_bbox, split_char_bbox = [], [], []
            start = 0
            if len(char_bbox) != len(text):
                char_bbox = [x_ for x in char_bbox for x_ in x]
            char_bbox = np.array(char_bbox).squeeze()
            for pos in split_pos:
                if not len(text[start: pos]):
                    import pdb;pdb.set_trace()               
                split_text.append(text[start: pos])
                x_min, y_min, x_max, y_max = np.min(char_bbox[start:pos][:,::2]), np.min(char_bbox[start:pos][:,1::2]), \
                                            np.max(char_bbox[start:pos][:,::2]), np.max(char_bbox[start:pos][:,1::2])
                split_bbox.append([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
                split_char_bbox.append(char_bbox[start:pos].tolist())
                start = pos
            return split_text, split_bbox, split_char_bbox

        def expand_list(data):
            return [[x] for x in data]

        def flatten_list(data):
            return [x_ for x in data for x_ in x]

        def delete_list(data, ids):
            ret = []
            for idx, data_ in enumerate(data):
                if idx not in ids:
                    ret.append(data_) 
            return ret

        texts, bboxes, char_bboxes = expand_list(texts), expand_list(bboxes), expand_list(char_bboxes)
        split_ids = np.random.choice(len(bboxes), min(len(bboxes) - 1, np.random.randint(max_split_num+1)), replace=False)
        for split_id in split_ids:
            if len(texts[split_id][0]) <= 1:
                continue
            texts[split_id], bboxes[split_id], char_bboxes[split_id] = \
                split_bboxes(texts[split_id][0], bboxes[split_id][0], char_bboxes[split_id])
        texts, bboxes, char_bboxes = flatten_list(texts), flatten_list(bboxes), flatten_list(char_bboxes)
        if max_merge_num > 0:
            merge_ids = np.random.choice(len(bboxes), min(len(bboxes) - 1, np.random.randint(max_merge_num+1)), replace=False)
            used = [0] * len(texts)
            delete_bboxes_ids = []
            for merge_id in merge_ids:
                texts, bboxes, same_line_bboxes_ids = merge_bboxes(merge_id, texts, bboxes, used)
                delete_bboxes_ids.extend(same_line_bboxes_ids)
            texts, bboxes = delete_list(texts, delete_bboxes_ids), delete_list(bboxes, delete_bboxes_ids)
        return texts, bboxes, char_bboxes

    def global_bbox_augment(self, bboxes, width, height):
        bboxes = np.array(bboxes)
        # scale: bboxes不变，width和height增大/减小
        x_min, y_min, x_max, y_max = np.min(bboxes[:, :, 0]), np.min(bboxes[:, :, 1]), np.max(bboxes[:, :, 0]), np.max(bboxes[:, :, 1])
        upper = 2
        width_lower, height_lower = max(0.5, x_max / width), max(0.5, y_max / height) # 保证scale后width和height的最小值>=bbox的最大值
        width_scale_ratio = random_truncnorm(mean=width_lower - 0.1, sigma=0.3, lower=width_lower, upper=upper)
        height_scale_ratio = random_truncnorm(mean=height_lower - 0.1, sigma=0.3, lower=height_lower, upper=upper)
        # print(f"width_scale_ratio: {width_scale_ratio}, height_scale_ratio: {height_scale_ratio}")
        width, height = int(width_scale_ratio * width), int(height_scale_ratio * height)

        # shift: bboxes向上下左右偏移
        x_shift_interval, y_shift_interval = [-x_min, max(-x_min + 1, width - x_max)], [-y_min, max(-y_min + 1, height - y_max)]
        try:
            x_shift, y_shift = np.random.randint(*x_shift_interval), np.random.randint(*y_shift_interval)
        except:
            x_shift, y_shift = 0, 0
        bboxes[:, :, 0] += x_shift
        bboxes[:, :, 1] += y_shift
        return bboxes, width, height

    # def cal_2d_pos_emb(self, bbox):
    #     position_coord_x = bbox[:, 0]
    #     position_coord_y = bbox[:, 3]
    #     rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
    #     rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
    #     rel_pos_x = self.relative_position_bucket(
    #         rel_pos_x_2d_mat,
    #         num_buckets=self.rel_2d_pos_bins,
    #         max_distance=self.max_rel_2d_pos,
    #     )
    #     rel_pos_y = self.relative_position_bucket(
    #         rel_pos_y_2d_mat,
    #         num_buckets=self.rel_2d_pos_bins,
    #         max_distance=self.max_rel_2d_pos,
    #     )
    #     rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size)
    #     rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size)
    #     rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
    #     rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
    #     rel_pos_x = rel_pos_x.contiguous()
    #     rel_pos_y = rel_pos_y.contiguous()
    #     rel_2d_pos = rel_pos_x + rel_pos_y
    #     return rel_2d_pos


    # def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    #     ret = 0
    #     if bidirectional:
    #         num_buckets //= 2
    #         ret += (relative_position > 0).long() * num_buckets
    #         n = torch.abs(relative_position)
    #     else:
    #         n = torch.max(-relative_position, torch.zeros_like(relative_position))
    #     # now n is in the range [0, inf)

    #     # half of the buckets are for exact increments in positions
    #     max_exact = num_buckets // 2
    #     is_small = n < max_exact

    #     # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    #     val_if_large = max_exact + (
    #             torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    #     ).to(torch.long)
    #     val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    #     ret += torch.where(is_small, n, val_if_large)
    #     return ret


def joint_height(height1, height2):
    assert len(height1) == 2, "len(height1): 2!={}".format(len(height1))
    assert len(height2) == 2, "len(height2): 2!={}".format(len(height2))
    min_h = min(height1[0], height2[0])
    max_h = max(height1[1], height2[1])
    h1 = height1[1] - height1[0]
    h2 = height2[1] - height2[0]
    joint = max(min(height1[1], height2[1]) - max(height1[0], height2[0]), 0)
    ratio = joint / (max_h - min_h + 1e-12)
    return ratio


def group_ocr_joint_height(ocr, hiou_threshold=0.5):
    group = {}
    for line in ocr:
        if 'ocr' in line:
            continue
        new_kk = (line["bbox"][1], line["bbox"][3])          
        max_hiou, max_kk = 0, None
        for kk, _ in group.items():
            hiou = joint_height(kk, new_kk)
            if hiou > hiou_threshold and hiou > max_hiou:
                max_hiou = hiou
                max_kk = kk
        if max_hiou > hiou_threshold:
            group[max_kk].append(line)
            update_kk = (min(max_kk[0], new_kk[0]), max(max_kk[1], new_kk[1]))
            group[update_kk] = group.pop(max_kk)
        else:
            if new_kk in group:
                group[new_kk].append(line)
            else:
                group[new_kk] = [line]
    for line in ocr:
        if 'ocr' in line:
            new_kk = (line["bbox"][1], line["bbox"][3])
            if new_kk in group:
                group[new_kk].append(line)
            else:
                group[new_kk] = [line]
    return group


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def points2polygon(points):
    """Convert k points to 1 polygon.

    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)
    points = points.flatten()
    if len(points) == 4:
        x1, y1, x2, y2 = points
        points = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)
    point_mat = points.reshape([-1, 2])
    return plg(point_mat)


class BBoxUtil:
    def __init__(self, with_subfield=True, with_sop=False) -> None:
        self.with_subfield = with_subfield
        self.with_sop = with_sop
        if self.with_sop:
            from docvqa.data.sop import SOPBert
            self.sop = SOPBert()
        else:
            self.sop = lambda x: 0.5
        if self.with_subfield:
            from docvqa.data.subfield import Subfield
            self.subfield = Subfield()
        else:
            self.subfield = lambda x: []

    @staticmethod
    def ocr_line_sort(ocr):
        for line in ocr:
            try:
                poly = [i for p in line["horizon_bndbox"] for i in p]
            except:
                import pdb;pdb.set_trace()
            line["horizon_bndbox"] = [
                min(poly[::2]),
                min(poly[1::2]),
                max(poly[::2]),
                max(poly[1::2]),
            ]
        ocr = sorted(ocr, key=lambda x: (x["horizon_bndbox"][1]))
        group = group_ocr_joint_height(ocr)
        group_list = sorted(list(group.items()), key=lambda x: x[0][0])
        group_list = [sorted(v, key=lambda x: x["horizon_bndbox"][0]) for k, v in group_list]
        ocr_sorted = [o for g in group_list for o in g]
        assert len(ocr_sorted) == len(ocr), "same length after ocr sort"
        return ocr_sorted

    @staticmethod
    def filter_ocr_within_group(ocr, group, ocr_used):
        group_poly = points2polygon(group)
        group_ocr = []
        for idx, item in enumerate(ocr):
            poly = points2polygon(item['bndbox'])
            if group_poly.is_valid and poly.is_valid and group_poly.area > 0 and poly.area > 0:
                if group_poly.intersects(poly):
                    inter_area = group_poly.intersection(poly).area
                else:
                    inter_area = 0
                if inter_area > 0 and inter_area / (poly.area + 1e-10) > 0.5 and not ocr_used[idx]:
                    group_ocr.append(item)
                    ocr_used[idx] = 1
        return group_ocr, ocr_used

    def sort_bbox_with_layout(self, ocr, doc_layout={}):
        if isinstance(doc_layout, str) and os.path.exists(doc_layout):
            try:
                doc_layout = load_json(doc_layout)
            except:
                doc_layout = []
        while 'resultMap' in doc_layout:
            doc_layout = doc_layout['resultMap']
        if 'layout' in doc_layout:
            doc_layout = doc_layout['layout']
        ocr_length = len(ocr)
        sub_layouts = sorted([layout for layout in doc_layout if layout['cls'] == 'sub'], key=lambda x:x['score'], reverse=True)

        ocr = self.ocr_group_line_sort(ocr, sub_layouts)
        classify_layouts = sorted([layout for layout in doc_layout if layout['cls'] != 'sub'], key=lambda x:x['score'], reverse=True)
        ocr = self.ocr_group_line_sort(ocr, classify_layouts, adjust_box=False)
        ocr = self.flatten_ocr(ocr)
        assert len(ocr) == ocr_length
        return ocr

    def ocr_group_line_sort(self, ocr, layouts, adjust_box=True):
        ocr_used = len(ocr) * [0]
        layouts_ocr = []        
        for layout_idx, layout in enumerate(layouts):
            layout = [round(x) for x in layout['bbox']]
            layout_ocr, ocr_used = self.filter_ocr_within_group(ocr, layout, ocr_used)
            if len(layout_ocr):
                layout_ocr = self.ocr_line_sort(layout_ocr)
                x1, y1, x2, y2 = layout
                layout = np.array([x1, y1, x2, y1, x2, y2, x1, y2]).reshape((4, 2))
                layout_ocr = self.flatten_ocr(layout_ocr)
                layouts_ocr.append({'horizon_bndbox': layout.tolist(), 'bndbox': layout.tolist(), 'ocr': layout_ocr, 'rotate_angle': 0})
        unused_ids = np.where(np.array(ocr_used) == 0)[0]
        layouts_ocr.extend([ocr[idx] for idx in unused_ids]) 
        layouts_ocr = self.ocr_line_sort(layouts_ocr)
        return layouts_ocr 

    def adjust_box2horizon(self, ocr, bbox_to_horizon=True):
        def box2horizon(box, rotateMat):
            box = np.array(box)
            if box.ndim == 1:
                x1, y1, x2, y2 = box
                box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            pt1, pt2, pt3, pt4 = np.array(box).reshape([4, 2])
            [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
            [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
            [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
            [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))
            # 处理反转的情况
            if pt2[1] > pt4[1]:
                pt2[1], pt4[1] = pt4[1], pt2[1]
            if pt1[0] > pt3[0]:
                pt1[0], pt3[0] = pt3[0], pt1[0]
            return np.array([pt1, pt2, pt3, pt4]).tolist()

        def poly2squard(poly):
            poly = np.array(poly).flatten()
            return [
                min(poly[::2]),
                min(poly[1::2]),
                max(poly[::2]),
                max(poly[1::2]),
            ]

        if not len(ocr):
            return ocr
        bboxes = np.array([x['bndbox'] for x in ocr])
        if 'content_pos_ori' in ocr[0]:
            char_bboxes = [x['content_pos_ori'] for x in ocr]
        else:
            char_bboxes = len(bboxes) * [[]]
        for line in ocr:
            line['rotate_angle'] = line.get('rotate_angle', self.cal_rotate_angle(line['bndbox']))
        angles = np.array([x['rotate_angle'] for x in ocr if 'content' in x and len(x['content']) > 1])
        if not len(angles):
            mean_angle = 0
        else:
            mean_angle = angles.mean()
        width, height = np.max(bboxes[:, ::2]) - np.min(bboxes[:, ::2]), np.max(bboxes[:, 1::2]) - np.min(bboxes[:, 1::2])
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), -mean_angle, 1)  # 按angle角度旋转获取旋转矩阵
        # 旋转后图像的四点坐标
        rotate_bboxes = []
        for idx, (box, char_bbox) in enumerate(zip(bboxes, char_bboxes)):
            if 'content' not in ocr[idx]: # 检测的layout是平行框，不需要做修正
                continue
            box = box2horizon(box, rotateMat)
            if len(char_bbox):
                char_bbox = [poly2squard(box2horizon(box, rotateMat)) for box in char_bbox]
            if bbox_to_horizon: # 改变bndbox和content_pos_ori
                ocr[idx]['horizon_bndbox'] = box
                ocr[idx]['bndbox'] = box
                if 'content_pos_ori' in ocr[idx]:
                    ocr[idx]['content_pos_ori'] = char_bbox
            else: # 不改变bndbox
                ocr[idx]['horizon_bndbox'] = box
        return ocr

    def flatten_ocr(self, ocr):
        flatten_ocr = []
        for item in ocr:
            if 'ocr' in item:
                flatten_ocr.extend(item['ocr'])
            else:
                flatten_ocr.append(item)
        return flatten_ocr

    def get_proposal_x(self, i, bboxes):
        if i >= len(bboxes):
            return []
        bbox_base = bboxes[i]
        text_base, position_base = bbox_base["content"], bbox_base["bbox"]
        proposals = []
        try:
            for j, bbox in enumerate(bboxes):
                text_j, position_j = bbox["content"], bbox["bbox"]
                if (
                    position_base[2] < position_j[0]
                    and joint_height(
                        (position_base[3], position_base[5]), (position_j[1], position_j[7])
                    )
                    > 0.5
                ):
                    proposals.append((j, bbox))
        except:
            import pdb;pdb.set_trace()
        if proposals:
            proposals = sorted(proposals, key=lambda x: x[1]["bbox"][0])[0]
        return proposals

    def get_proposal_y(self, i, bboxes):
        if i >= len(bboxes):
            return []
        bbox_base = bboxes[i]
        text_base, position_base = bbox_base["content"], bbox_base["bbox"]
        proposals = []
        for j, bbox in enumerate(bboxes):
            text_j, position_j = bbox["content"], bbox["bbox"]
            if (
                position_base[7] < position_j[1]
                and joint_height(
                    (position_base[6], position_base[4]), (position_j[0], position_j[2])
                )
                > 0.5
            ):
                proposals.append((j, bbox))
        if proposals:
            proposals = sorted(proposals, key=lambda x: x[1]["bbox"][1])[0]
        return proposals

    def bbox_align(self, bbox1, bbox2):
        pass

    def sop_lm(self, text1, text2):
        ppl1 = self.sop(text1)
        ppl2 = self.sop(text2)
        if ppl1 > ppl2:
            return True
        else:
            return False

    def subfield_det(self, image_raw):
        image = self.raw2image(image_raw)
        result = self.subfield(image)
        return result

    def find_components(self, line_edges, num_line):
        line_list = []
        passed = set()
        graph = nx.Graph()
        graph.add_edges_from(line_edges)
        for nodes in nx.connected_components(graph):
            line_list.append(nodes)
            passed |= nodes
        for i in range(num_line):
            if i not in passed:
                line_list.append({i})
        return line_list

    @staticmethod
    def cal_rotate_angle(box):
        def order_points(vertices):
            assert vertices.ndim == 2
            assert vertices.shape[-1] == 2
            N = vertices.shape[0]
            if N == 0:
                return vertices

            center = np.mean(vertices, axis=0)
            directions = vertices - center
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            sort_idx = np.argsort(angles)
            vertices = vertices[sort_idx]

            left_top = np.min(vertices, axis=0)
            dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
            lefttop_idx = np.argmin(dists)
            indexes = (np.arange(N, dtype=np.int64) + lefttop_idx) % N
            return vertices[indexes]

        box = np.reshape(box, [4, 2])
        box = order_points(box)
        pt1, pt2, pt3, pt4 = np.reshape(box, [4, 2])
        heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
        angle = math.asin((pt1[1] - pt2[1]) / (heightRect + 1e-10)) * (180 / math.pi)  # 矩形框旋转角度
        return angle

    def sort_bbox(self, ocr, layout=None, image_raw=None, sort_method="position_only", bbox_to_horizon=True):
        ocr = self.parse_ocr(ocr)
        # for line in ocr:
        #     line['rotate_angle'] = self.cal_rotate_angle(line['bndbox'])

        ocr = self.adjust_box2horizon(ocr, bbox_to_horizon)
        if sort_method == "layout_sort":
            if layout is not None:
                return self.sort_bbox_with_layout(ocr, layout)
            else:
                return self.ocr_line_sort(ocr)
        if sort_method == "position_only":
            return self.ocr_line_sort(ocr)
        elif sort_method == "subfield":
            return self.sort_bbox_with_subfield(ocr, image_raw)
        elif sort_method == "subfield_sop":
            return self.sort_bbox_with_subfield_sop(ocr, image_raw)
        elif sort_method == "simple":
            return self.sort_bbox_with_simple(ocr)
        else:
            return self.sort_bbox_with_nosort(ocr)

    def sort_bbox_with_nosort(self, ocr, image_raw=None):
        ocr = self.parse_ocr(ocr)
        return ocr

    def sort_bbox_with_simple(self, ocr, image_raw=None):
        ocr = self.parse_ocr(ocr)
        ocr = self.ocr_line_sort(ocr)
        return ocr

    def sort_bbox_with_position_only(self, ocr, image_raw=None):
        ocr = self.parse_ocr(ocr)
        ocr = self.ocr_line_sort(ocr)

        line_edges = []
        for i in range(len(ocr)):
            target_x = self.get_proposal_x(i, ocr)
            if target_x:
                line_edges.append((i, target_x[0]))

        line_list = self.find_components(line_edges, len(ocr))

        group_list = [[ocr[i] for i in line] for line in line_list]
        group_list = [sorted(group, key=lambda x: x["bbox"][0]) for group in group_list]
        group_list = sorted(group_list, key=lambda x: x[0]["bbox"][1])
        ocr_sorted = [o for g in group_list for o in g]
        assert len(ocr_sorted) == len(ocr), "same length after ocr sort"

        return ocr_sorted

    def iou(self, boxA, boxB):
        assert len(boxA) == 4, "len(boxA): 4!={}".format(len(boxA))
        assert len(boxB) == 4, "len(boxB): 4!={}".format(len(boxB))

        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        iou_ratio = interArea / max(min(boxAArea, boxBArea), 1)
        return iou_ratio

    def sort_bbox_sop(self, ocr, group_list):
        ocr_group_cnt = [len(g) for g in group_list for o in g]
        for i in range(len(ocr)):
            if ocr_group_cnt[i] >= 2:
                target_y = self.get_proposal_y(i, ocr)
                if target_y and i + 1 < len(ocr):
                    text_i, position_i = ocr[i]["content"], ocr[i]["bbox"]
                    text_i1, position_i1 = ocr[i + 1]["content"], ocr[i + 1]["bbox"]
                    text_y, position_y = target_y[1]["content"], target_y[1]["bbox"]
                    if (
                        self.sop_lm(
                            "{}{}".format(text_i, text_i1),
                            "{}{}".format(text_i, text_y),
                        )
                        and position_i1[7] < position_y[3]
                        and position_i1[6] > position_y[2]
                    ):
                        target_bbox = ocr.pop(target_y[0])
                        ocr = ocr[: (i + 1)] + [target_bbox] + ocr[(i + 1) :]
                        break
        return ocr

    def subfield_sub_sort(self, ocr):
        line_edges = []
        for i in range(len(ocr)):
            target_x = self.get_proposal_x(i, ocr)
            if target_x:
                line_edges.append((i, target_x[0]))

        line_list = self.find_components(line_edges, len(ocr))

        group_list = [[ocr[i] for i in line] for line in line_list]
        group_list = [sorted(group, key=lambda x: x["bbox"][0]) for group in group_list]
        group_list = sorted(group_list, key=lambda x: x[0]["bbox"][1])
        return group_list

    def sort_bbox_with_subfield(self, ocr, image_raw=None):
        subfield_rect = []
        if image_raw is not None and self.with_subfield:
            subfield_rect = self.subfield_det(image_raw)
            subfield_rect = subfield_rect.tolist()

        ocr = self.parse_ocr(ocr)
        ocr = self.ocr_line_sort(ocr)

        subfields_dict, no_subfields = dict(), []
        for i in range(len(ocr)):
            rect = ocr[i]["bbox4"]
            isin_subfield = False
            for sf_rect in subfield_rect:
                if self.iou(rect, sf_rect) > 0.5:
                    if tuple(sf_rect) in subfields_dict:
                        subfields_dict[tuple(sf_rect)].append(ocr[i])
                    else:
                        subfields_dict[tuple(sf_rect)] = [ocr[i]]
                    isin_subfield = True
                    break
            if not isin_subfield:
                no_subfields.append(ocr[i])

        group_list = self.subfield_sub_sort(no_subfields)
        for _, subfield_vlist in subfields_dict.items():
            subfield_vlist = self.subfield_sub_sort(subfield_vlist)
            subfield_vlist = [o for g in subfield_vlist for o in g]
            group_list.append(subfield_vlist)

        group_list = sorted(group_list, key=lambda x: x[0]["bbox"][1])
        ocr_sorted = [o for g in group_list for o in g]
        assert len(ocr_sorted) == len(ocr), "same length after ocr sort"
        return ocr_sorted

    def sort_bbox_with_subfield_sop(self, ocr, image_raw=None):
        subfield_rect = []
        if image_raw is not None and self.with_subfield:
            subfield_rect = self.subfield_det(image_raw)
            subfield_rect = subfield_rect.tolist()

        ocr = self.parse_ocr(ocr)
        ocr = self.ocr_line_sort(ocr)

        subfields_dict, no_subfields = dict(), []
        for i in range(len(ocr)):
            rect = ocr[i]["bbox4"]
            isin_subfield = False
            for sf_rect in subfield_rect:
                if self.iou(rect, sf_rect) > 0.5:
                    if tuple(sf_rect) in subfields_dict:
                        subfields_dict[tuple(sf_rect)].append(ocr[i])
                    else:
                        subfields_dict[tuple(sf_rect)] = [ocr[i]]
                    isin_subfield = True
                    break
            if not isin_subfield:
                no_subfields.append(ocr[i])

        group_list = self.subfield_sub_sort(no_subfields)
        for _, subfield_vlist in subfields_dict.items():
            subfield_vlist = self.subfield_sub_sort(subfield_vlist)
            subfield_vlist = [o for g in subfield_vlist for o in g]
            group_list.append(subfield_vlist)

        group_list = sorted(group_list, key=lambda x: x[0]["bbox"][1])
        ocr_sorted = [o for g in group_list for o in g]

        ocr_sorted = self.sort_bbox_sop(ocr_sorted, group_list)

        assert len(ocr_sorted) == len(ocr), "same length after ocr sort"
        return ocr_sorted

    def read_json(self, json_file):
        assert os.path.exists(json_file), "{} is no exist".format(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def dump_json(self, json_dict, json_file):
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)

    def read_image(self, image_file):
        assert os.path.exists(image_file), "{} is no exist".format(image_file)
        with open(image_file, "rb") as f:
            data = f.read()
        data = base64.b64encode(data).decode("utf-8")
        return data

    def raw2image(self, imageRaw):
        binary = base64.b64decode(imageRaw.encode("utf-8"))
        image = Image.open(io.BytesIO(binary)).convert("RGB")
        image = np.asarray(image)
        return image

    def parse_ocr(self, ocr):
        if isinstance(ocr, dict) and "data" in ocr:
            ocr = ocr["data"]
        if isinstance(ocr, dict) and "result" in ocr:
            ocr = ocr["result"]
        if isinstance(ocr, dict) and "result" in ocr:
            ocr = ocr["result"]

        if isinstance(ocr, dict) and "resultMap" in ocr:
            ocr = ocr["resultMap"]
        if isinstance(ocr, dict) and "result" in ocr:
            ocr = ocr["result"]

        if not isinstance(ocr, list):
            return []

        if isinstance(ocr, list) and len(ocr) > 0 and isinstance(ocr[0], list):
            flat_ocr = []
            for cell in ocr:
                flat_ocr.extend(cell)
            ocr = flat_ocr

        for line in ocr:
            poly = [i for p in line["bndbox"] for i in p]
            line["bbox"] = poly

        return ocr

    def do_sort_bbox(
        self, image_path, ocr_path, ocr_path_sorted, sort_method="position_only"
    ):
        os.makedirs(ocr_path_sorted, exist_ok=True)
        for image_name in os.listdir(image_path):
            try:
                if image_name.startswith("."):
                    continue
                image_file = os.path.join(image_path, image_name)
                ocr_file = os.path.join(
                    ocr_path, "{}.json".format(image_name.rsplit(".", 1)[0])
                )
                ocr_sorted_file = os.path.join(
                    ocr_path_sorted, "{}.json".format(image_name.rsplit(".", 1)[0])
                )

                image_raw = self.read_image(image_file)
                ocr = self.read_json(ocr_file)

                ocr = self.sort_bbox(ocr, image_raw, sort_method=sort_method)

                ocr = {"data": {"result": {"result": ocr}}}

                self.dump_json(ocr, ocr_sorted_file)

            except Exception as e:
                print(image_name, e)
                continue





def random_truncnorm(mean, sigma, lower, upper):
    import scipy.stats as stats
    X = stats.truncnorm((lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
    return X.rvs(1)[0]



