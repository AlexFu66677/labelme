
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import PIL.ImageDraw
from tqdm import tqdm

label_to_num = {}
categories_list = []
labels_list = []


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def images_labelme(data, num):
    image = {}
    image['height'] = data['imageHeight']
    image['width'] = data['imageWidth']
    image['id'] = num + 1
    if '\\' in data['imagePath']:
        image['file_name'] = data['imagePath'].split('\\')[-1]
    else:
        image['file_name'] = data['imagePath'].split('/')[-1]
    return image

def categories(label, labels_list):
    category = {}
    category['supercategory'] = 'component'
    category['id'] = len(labels_list) + 1
    category['name'] = label
    return category


def annotations_rectangle(points, label, image_num, object_num, label_to_num):
    annotation = {}
    seg_points = np.asarray(points).copy()
    seg_points[1, :] = np.asarray(points)[2, :]
    seg_points[2, :] = np.asarray(points)[1, :]
    annotation['segmentation'] = [list(seg_points.flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = list(
        map(float, [
            points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                1] - points[0][1]
        ]))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def annotations_polygon(height, width, points, label, image_num, object_num,
                        label_to_num):
    annotation = {}
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = list(map(float, get_bbox(height, width, points)))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def get_bbox(height, width, points):
    polygons = points
    mask = np.zeros([height, width], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [
        left_top_c, left_top_r, right_bottom_c - left_top_c,
        right_bottom_r - left_top_r
    ]


def deal_json(res,ds_type, img_path, json_path):
    data_coco = {}
    images_list = []
    annotations_list = []
    image_num = -1
    object_num = -1
    for img_file in os.listdir(img_path):
        img_label = os.path.splitext(img_file)[0]
        if img_file.split('.')[
                -1] not in ['bmp', 'jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG']:
            continue
        label_file = osp.join(json_path, img_label + '.json')
        res.append(('Generating dataset from:', label_file))
        image_num = image_num + 1
        with open(label_file) as f:
            data = json.load(f)
            if ds_type == 'labelme':
                images_list.append(images_labelme(data, image_num))
            if ds_type == 'labelme':
                for shapes in data['shapes']:
                    object_num = object_num + 1
                    label = shapes['label']
                    if label not in labels_list:
                        categories_list.append(categories(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    p_type = shapes['shape_type']
                    if p_type == 'polygon':
                        points = shapes['points']
                        annotations_list.append(
                            annotations_polygon(data['imageHeight'], data[
                                'imageWidth'], points, label, image_num,
                                                object_num, label_to_num))

                    if p_type == 'rectangle':
                        (x1, y1), (x2, y2) = shapes['points']
                        x1, x2 = sorted([x1, x2])
                        y1, y2 = sorted([y1, y2])
                        points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]
                        annotations_list.append(
                            annotations_rectangle(points, label, image_num,
                                                  object_num, label_to_num))
    data_coco['images'] = images_list
    data_coco['categories'] = categories_list
    data_coco['annotations'] = annotations_list
    return data_coco

def Generator(folder_data,value_data):

    res=[]
    res_str = ""
    json_input_dir=folder_data[0]
    image_input_dir=folder_data[1]
    output_dir=folder_data[2]
    train_proportion=float(value_data[0])
    val_proportion=float(value_data[1])
    test_proportion=float(value_data[2])
    dataset_type = 'labelme'

    total_num = len(glob.glob(osp.join(json_input_dir, '*.json')))
    if train_proportion != 0:
        train_num = int(total_num * train_proportion)
        out_dir =output_dir + '/train'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        train_num = 0
    if val_proportion == 0.0:
        val_num = 0
        test_num = total_num - train_num
        out_dir = output_dir + '/test'
        if test_proportion != 0.0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        val_num = int(total_num * val_proportion)
        test_num = total_num - train_num - val_num
        val_out_dir =output_dir + '/val'
        if not os.path.exists(val_out_dir):
            os.makedirs(val_out_dir)
        test_out_dir = output_dir + '/test'
        if test_proportion != 0.0 and not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
    count = 1
    for img_name in os.listdir(image_input_dir):
        if count <= train_num:
            if osp.exists(output_dir + '/train/'):
                shutil.copyfile(
                    osp.join(image_input_dir, img_name),
                    osp.join(output_dir + '/train/', img_name))
        else:
            if count <= train_num + val_num:
                if osp.exists(output_dir + '/val/'):
                    shutil.copyfile(
                        osp.join(image_input_dir, img_name),
                        osp.join(output_dir + '/val/', img_name))
            else:
                if osp.exists(output_dir + '/test/'):
                    shutil.copyfile(
                        osp.join(image_input_dir, img_name),
                        osp.join(output_dir + '/test/', img_name))
        count = count + 1

    # Deal with the json files.
    if not os.path.exists(output_dir + '/annotations'):
        os.makedirs(output_dir + '/annotations')
    if train_proportion != 0:
        train_data_coco = deal_json(res,dataset_type,output_dir + '/train',json_input_dir)
        train_json_path = osp.join(output_dir + '/annotations','instance_train.json')
        json.dump(
            train_data_coco,
            open(train_json_path, 'w'),
            indent=4,
            cls=MyEncoder)
    if val_proportion != 0:
        val_data_coco = deal_json(res,dataset_type,
                                  output_dir + '/val',
                                  json_input_dir)
        val_json_path = osp.join(output_dir + '/annotations',
                                 'instance_val.json')
        json.dump(
            val_data_coco,
            open(val_json_path, 'w'),
            indent=4,
            cls=MyEncoder)
    if test_proportion != 0:
        test_data_coco = deal_json(res,dataset_type,
                                   output_dir + '/test',
                                   json_input_dir)
        test_json_path = osp.join(output_dir + '/annotations',
                                  'instance_test.json')
        json.dump(
            test_data_coco,
            open(test_json_path, 'w'),
            indent=4,
            cls=MyEncoder)
        for i in range(len(res)):
             res_str=res_str+'\n'+str(res[i])
    return res_str
# Generator('C:/Users/fjl\Desktop\data/anno','C:/Users/fjl/Desktop/data/img','C:/Users/fjl\Desktop\data',0.4,0.3,0.3)