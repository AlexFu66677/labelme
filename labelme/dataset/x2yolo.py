import json
import os
import glob
import os.path as osp
import shutil
import random
from qtpy import QtWidgets


def convert(img_size, box):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)



def decode_json(json_path, output_dir, name_list, res, sequence=None):
    data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']
    file = open(output_dir, 'w')

    for image in data['shapes']:
        label_name = image['label']

        if sequence:  # If sequence is provided
            if label_name not in sequence:
                error_message = f"Label '{label_name}' not found in the provided sequence!"
                show_error_message(error_message)
                return error_message
            else:
                label_index = sequence.index(label_name)
                name_list[label_name] = label_index
        else:
            if not name_list:
                name_list.setdefault(label_name, 0)
            elif label_name not in name_list:
                name_list[label_name] = len(name_list)

        if (image['shape_type'] == 'rectangle' or (image['shape_type'] == 'mask' and len(image['points']) == 2)):
            x1 = int(image['points'][0][0])
            y1 = int(image['points'][0][1])
            x2 = int(image['points'][1][0])
            y2 = int(image['points'][1][1])
            points = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), points)
            file.write(str(name_list[label_name]) + " " + " ".join([str(point) for point in bbox]) + '\n')
        else:
            return f"Image '{json_path}' not support {image['shape_type']}"
    file.close()
    # res.append(('Generating dataset from:' + json_path))
    return 'over'


def show_error_message(error_message):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText("Error")
    msg_box.setInformativeText(error_message)
    msg_box.setWindowTitle("Error")
    msg_box.exec_()


def YoloGenerator(folder_data, value_data, sequence=None):
    res = []
    res_str = ""
    json_input_dir = folder_data[0]
    image_input_dir = folder_data[1]
    output_dir = folder_data[2]
    if value_data[0]:
        train_proportion = float(value_data[0])
    else:
        train_proportion = 0
    if value_data[1]:
        test_proportion = float(value_data[1])
    else:
        test_proportion = 0
    if value_data[2]:
        val_proportion = float(value_data[2])
    else:
        val_proportion = 0

    try:
        assert os.path.exists(json_input_dir)
    except AssertionError:
        return 'The json folder does not exist!'
    try:
        assert os.path.exists(image_input_dir)
    except AssertionError:
        return 'The image folder does not exist!'
    try:
        assert abs(train_proportion + val_proportion + test_proportion - 1.0) < 1e-5
    except AssertionError:
        return 'The sum of proportions of training, validation and test datasets must be 1!'

    total_num = len(glob.glob(osp.join(json_input_dir, '*.json')))
    if train_proportion != 0:
        train_num = int(total_num * train_proportion)
        train_image_out_dir = output_dir + '/train'
        train_label_out_dir = output_dir + '/train'
        os.makedirs(train_image_out_dir, exist_ok=True)
        os.makedirs(train_label_out_dir, exist_ok=True)
    else:
        train_num = 0
    if val_proportion == 0.0:
        val_num = 0
        test_num = total_num - train_num
        test_image_out_dir = output_dir + '/test'
        test_label_out_dir = output_dir + '/test'
        if test_proportion != 0.0:
            os.makedirs(test_image_out_dir, exist_ok=True)
            os.makedirs(test_label_out_dir, exist_ok=True)
    else:
        val_num = int(total_num * val_proportion)
        test_num = total_num - train_num - val_num
        val_image_out_dir = output_dir + '/val'
        val_label_out_dir = output_dir + '/val'
        os.makedirs(val_image_out_dir, exist_ok=True)
        os.makedirs(val_label_out_dir, exist_ok=True)
        test_image_out_dir = output_dir + '/test'
        test_label_out_dir = output_dir + '/test'
        if test_proportion != 0.0:
            os.makedirs(test_image_out_dir, exist_ok=True)
            os.makedirs(test_label_out_dir, exist_ok=True)

    count = 1
    img_names = []
    images_input = os.listdir(image_input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    json_file_names = {os.path.splitext(f)[0] for f in os.listdir(json_input_dir) if f.endswith('.json')}
    image_file_names = [
        f for f in images_input
        if os.path.splitext(f)[-1].lower() in image_extensions and os.path.splitext(f)[0] in json_file_names
    ]
    for file in image_file_names:
        if os.path.isfile(os.path.join(image_input_dir, file)):
            img_names.append(str(file))
    random.shuffle(img_names)
    for img_name in img_names:
        if count <= train_num:
            if osp.exists(train_image_out_dir):
                shutil.copyfile(
                    osp.join(image_input_dir, img_name),
                    osp.join(train_image_out_dir, img_name))
        else:
            if count <= train_num + val_num:
                if osp.exists(val_image_out_dir):
                    shutil.copyfile(
                        osp.join(image_input_dir, img_name),
                        osp.join(val_image_out_dir, img_name))
            else:
                if osp.exists(test_image_out_dir):
                    shutil.copyfile(
                        osp.join(image_input_dir, img_name),
                        osp.join(test_image_out_dir, img_name))
        count = count + 1

    # Deal with the json files.
    name_list = {}

    # Convert sequence string to list of labels
    if sequence:
        sequence = sequence.split()

    if train_proportion != 0:
        train_image_names = os.listdir(train_image_out_dir)
        for train_image_name in train_image_names:
            json_file_name = os.path.splitext(os.path.basename(train_image_name))[0]
            train_json_name = osp.join(json_input_dir, json_file_name + '.json')
            train_label_output = osp.join(train_label_out_dir, json_file_name + '.txt')
            res = decode_json(train_json_name, train_label_output, name_list, res, sequence)
            if res!="over":
                return res

    if val_proportion != 0:
        val_image_names = os.listdir(val_image_out_dir)
        for val_image_name in val_image_names:
            json_file_name = os.path.splitext(os.path.basename(val_image_name))[0]
            val_json_name = osp.join(json_input_dir, json_file_name + '.json')
            val_label_output = osp.join(val_label_out_dir, json_file_name + '.txt')
            res = decode_json(val_json_name, val_label_output, name_list, res, sequence)
            if res!="over":
                return res

    if test_proportion != 0:
        test_image_names = os.listdir(test_image_out_dir)
        for test_image_name in test_image_names:
            json_file_name = os.path.splitext(os.path.basename(test_image_name))[0]
            test_json_name = osp.join(json_input_dir, json_file_name + '.json')
            test_label_output = osp.join(test_label_out_dir, json_file_name + '.txt')
            res = decode_json(test_json_name, test_label_output, name_list, res, sequence)
            if res!="over":
                return res

    # for i in range(len(res)):
    #     res_str = res_str + '\n' + str(res[i])
    # return res_str
    return res

# Usage example
# app = QtWidgets.QApplication([])  # This is needed to create a QApplication context
# folder_data = ['C:/Users/fjl/Desktop/dataset/json', 'C:/Users/fjl/Desktop/dataset/image', 'C:/Users/fjl/Desktop/dataset']
# value_data = [0.4, 0.3, 0.3]
# sequence = "label1 label2 label3"
# result = YoloGenerator(folder_data, value_data, sequence)
# print(result)
# app.exec_()
