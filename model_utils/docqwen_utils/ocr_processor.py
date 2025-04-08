# -*- encoding: utf-8 -*-
import json
import os
import tqdm
import numpy as np
import cv2
import math
from typing import Any
import yaml
from collections import defaultdict
from .bbox_processor import BboxProcessor, BBoxUtil
from copy import deepcopy
from functools import partial
from .util import load_json, load_jsonl, dump_json, dump_jsonl


class OCRProcessor:
    def __init__(self, ocr_dir=None, remove_blank=True, bbox_to_horizon=False):
        self.ocr_dir = ocr_dir
        self.remove_blank = remove_blank
        self.bbu = BBoxUtil(with_subfield=False, with_sop=False)
        self.bbox_to_horizon = bbox_to_horizon
        self.bbox_processor = BboxProcessor()

    def load_ocr(self):
        if self.ocr_dir is None:
            return None
        if isinstance(self.ocr_dir, dict):
            return self.ocr_dir
        print('loading ocr...')
        ocrs = {}
        if os.path.isdir(self.ocr_dir):
            ocr_files = [os.path.join(self.ocr_dir, x) for x in os.listdir(self.ocr_dir) if 'update_angle_0410' in x]
        else:
            ocr_files = [self.ocr_dir]
        np.random.shuffle(ocr_files)
        for ocr_path in ocr_files:
            ocr = load_json(ocr_path)
            for imgname, ocr_ in ocr.items():
                if 'ocr_results' in ocr_:
                    ocr_ = ocr_['ocr_results']
                ocrs[imgname] = ocr_
        # for imgname in list(ocrs.keys()):
        #     ocrs[os.path.basename(imgname)] = ocrs.pop(imgname)
        return ocrs 

    def preprocess(self, ocr, sort=False):
        ocr_text, ocr_bbox, ocr_char_bbox = [], [], []
        if 'ocr_results' in ocr:
            ocr = ocr['ocr_results']
        if sort:
            ocr = self.bbu.sort_bbox(ocr, layout=None, sort_method='position_only', bbox_to_horizon=self.bbox_to_horizon)
        for item in ocr:
            if len(item['content'].strip()) > 0:
                ocr_text.append(item['content'])
                ocr_bbox.append(item['bndbox'])
                content_pos_key = 'content_pos_resize' if 'content_pos_resize' in item else 'content_pos_ori'
                ocr_char_bbox.append(item.get(content_pos_key, self.generate_content_pos_resize(item['content'], item['bndbox'])))
        if self.remove_blank:
            ocr_text, ocr_bbox, ocr_char_bbox = self.remove_text_blank(ocr_text, ocr_bbox, ocr_char_bbox)
        ocr_text = [self.remove_special_token(text) for text in ocr_text]
        # filter empty text
        ocr_text_bbox = [(x, y, z) for x, y, z in zip(ocr_text, ocr_bbox, ocr_char_bbox) if len(x.strip())]
        ocr_text, ocr_bbox, ocr_char_bbox = [x[0] for x in ocr_text_bbox], [x[1] for x in ocr_text_bbox], [x[2] for x in ocr_text_bbox]
        return ocr_text, ocr_bbox, ocr_char_bbox

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
            indexes = (np.arange(N, dtype=np.int) + lefttop_idx) % N
            return vertices[indexes]

        box = np.reshape(box, [4, 2])
        box = order_points(box)
        pt1, pt2, pt3, pt4 = np.reshape(box, [4, 2])
        heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
        angle = math.asin((pt1[1] - pt2[1]) / (heightRect + 1e-10)) * (180 / math.pi)  # 矩形框旋转角度
        return angle

    def adjust_box2horizon(self, ocr):
        if not len(ocr):
            return ocr
        bboxes = np.array([x['bndbox'] for x in ocr])
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
        for idx, box in enumerate(bboxes):
            if 'content' not in ocr[idx]: # 检测的layout是平行框，不需要做修正
                continue
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
            ocr[idx]['bndbox'] = np.array([pt1, pt2, pt3, pt4]).tolist()
        return ocr

    def augment(self, ocr_text, ocr_bbox, ocr_char_bbox):
        ocr_bbox = np.array(ocr_bbox)
        box_max_h = np.abs(ocr_bbox[:, 0, 1] - ocr_bbox[:, 1, 1]).max()
        x_max, y_max = np.max(ocr_bbox[:, :, 0]), np.max(ocr_bbox[:, :, 1])
        width, height = x_max * (1 + np.random.uniform()), y_max * (1 + np.random.uniform()) # width, height 为图像的宽高，模拟图像随机缩放
        height = int(max(box_max_h * 20, height))
        ocr_text, ocr_bbox, ocr_char_bbox = self.bbox_processor.random_split_merge(ocr_text, ocr_bbox, ocr_char_bbox, max_merge_num=0)
        try:
            ocr_bbox, width, height = self.bbox_processor.global_bbox_augment(ocr_bbox, width, height)
        except Exception as E:
            print(E)   
        # filter empty text
        assert len(ocr_text) == len(ocr_bbox) == len(ocr_char_bbox)
        ocr_text_bbox = [(x, y, z) for x, y, z in zip(ocr_text, ocr_bbox, ocr_char_bbox) if len(x.strip())]
        ocr_text, ocr_bbox, ocr_char_bbox = [x[0] for x in ocr_text_bbox], [x[1] for x in ocr_text_bbox], [x[2] for x in ocr_text_bbox]
        return ocr_text, ocr_bbox, ocr_char_bbox, width, height

    def box_norm(self, ocr_bbox, width=None, height=None):
        if width is None or height is None:
            ocr_bbox = np.array(ocr_bbox)
            box_max_h = np.abs(ocr_bbox[:, 0, 1] - ocr_bbox[:, 1, 1]).max()
            x_max, y_max = np.max(ocr_bbox[:, :, 0]), np.max(ocr_bbox[:, :, 1])
            width, height = x_max * 1.5, y_max * 1.5 # width, height 为图像的宽高，模拟图像随机缩放
        ocr_bbox = np.array([self.bbox_processor.box_norm(box, width=width, height=height) for box in ocr_bbox])  
        return ocr_bbox  

    @staticmethod
    def generate_content_pos_resize(content, bndbox):
        bndbox = np.array(bndbox)
        char_width = (abs(bndbox[1,0] - bndbox[0,0]) + abs(bndbox[2,0] - bndbox[3,0])) / (2 * len(content))
        char_height = (abs(bndbox[3,1] - bndbox[0,1]) + abs(bndbox[2,1] - bndbox[1,1])) / 2
        delta_h = ((bndbox[0,1] - bndbox[1,1]) + (bndbox[3,1] - bndbox[2,1])) / (2 * len(content))
        x0, y0 = bndbox[0]
        content_pos_resize = []
        for i in range(len(content)):
            char_box = [x0+char_width*i, y0+delta_h*i, x0+char_width*(i+1), y0+delta_h*(i+1)]
            content_pos_resize.append(char_box)
        content_pos_resize = np.maximum(content_pos_resize, 0).tolist()
        assert len(content_pos_resize) == len(content), f"content_pos_resize number: {len(content_pos_resize)}, while content length: {len(content)}"
        return content_pos_resize

    @staticmethod
    def remove_text_blank(ocr_text, ocr_bbox, ocr_char_bbox):
        def get_ocr_bbox(char_bbox):
            x1, y1 = char_bbox[0][0], char_bbox[0][1]
            x2, y2 = char_bbox[-1][2], char_bbox[-1][1]
            x3, y3 = char_bbox[-1][2], char_bbox[-1][3]          
            x4, y4 = char_bbox[0][0], char_bbox[0][3]      
            return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  

        def is_chinese(text):
            '''判断是否存在中文字符'''
            for cha in text:
                if '\u0e00' <= cha <= '\u9fa5':
                    return True
            return False
                    
        new_ocr_text, new_ocr_bbox, new_ocr_char_bbox = [], [], []
        for text, bbox, char_bbox in zip(ocr_text, ocr_bbox, ocr_char_bbox):
            text = text.strip()
            if is_chinese(text) and ' ' in text and np.random.uniform() < 0.8:
                if np.random.uniform() < 0.5: # 删除空格
                    text = text.replace(' ', '')
                    char_bbox = char_bbox[:len(text)]
                    bbox = get_ocr_bbox(char_bbox)
                    new_ocr_text.append(text)
                    new_ocr_bbox.append(bbox)
                    new_ocr_char_bbox.append(char_bbox)
                else: # 分成n个框
                    tail_text = text
                    tail_char_bbox = char_bbox
                    while ' ' in tail_text:
                        blank_start_id = tail_text.index(' ')
                        blank_end_id = blank_start_id + 1
                        while blank_end_id < len(tail_text) and tail_text[blank_end_id] == ' ':
                            blank_end_id += 1
                        text, tail_text = tail_text[:blank_start_id+1], tail_text[blank_end_id:]
                        char_bbox, tail_char_bbox = tail_char_bbox[:blank_start_id+1], tail_char_bbox[blank_end_id:]
                        bbox = get_ocr_bbox(char_bbox)
                        new_ocr_text.append(text)
                        new_ocr_bbox.append(bbox)
                        new_ocr_char_bbox.append(char_bbox)
                    new_ocr_text.append(tail_text)
                    new_ocr_bbox.append(get_ocr_bbox(tail_char_bbox))
                    new_ocr_char_bbox.append(tail_char_bbox)
            else:
                new_ocr_text.append(text)
                new_ocr_bbox.append(bbox)
                new_ocr_char_bbox.append(char_bbox)
        return new_ocr_text, new_ocr_bbox, new_ocr_char_bbox

    @staticmethod
    def remove_special_token(ocr_string):
        for sp_token in ['[MASK]', '[gMASK]']:
            while sp_token in ocr_string:
                sp_start_id = ocr_string.index(sp_token)
                sp_end_id = sp_start_id + len(sp_token)
                ocr_string = ocr_string[:sp_start_id] + ocr_string[sp_end_id:]
        return ocr_string

