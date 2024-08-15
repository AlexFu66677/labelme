import json
import os
import os.path as osp
import shutil
import chardet
import glob
import random


def get_encoding(path):
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    f.close()
    return file_encoding


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


def convert(image_dir, json_dir, dataset_save_dir, name_list, res):
    new_image_dir = osp.join(dataset_save_dir, "images")
    os.makedirs(new_image_dir)

    images_input = os.listdir(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    json_file_names = {os.path.splitext(f)[0] for f in os.listdir(json_dir) if f.endswith('.json')}
    img_names = [
        f for f in images_input
        if os.path.splitext(f)[-1].lower() in image_extensions and os.path.splitext(f)[0] in json_file_names
    ]
    for img_name in img_names:
        if is_pic(img_name):
            shutil.copyfile(
                osp.join(image_dir, img_name),
                osp.join(new_image_dir, img_name))
    xml_dir = osp.join(dataset_save_dir, "annotations")
    os.makedirs(xml_dir)
    json2xml(new_image_dir, json_dir, xml_dir, name_list, res)


def json2xml(image_dir, json_dir, xml_dir, name_list, res):
    import xml.dom.minidom as minidom
    i = 0
    for img_name in os.listdir(image_dir):
        img_name_part = osp.splitext(img_name)[0]
        json_file = osp.join(json_dir, img_name_part + ".json")
        i += 1
        if not osp.exists(json_file):
            os.remove(osp.join(image_dir, img_name))
            continue
        xml_doc = minidom.Document()
        root = xml_doc.createElement("annotation")
        xml_doc.appendChild(root)
        node_folder = xml_doc.createElement("folder")
        node_folder.appendChild(xml_doc.createTextNode("images"))
        root.appendChild(node_folder)
        node_filename = xml_doc.createElement("filename")
        node_filename.appendChild(xml_doc.createTextNode(img_name))
        root.appendChild(node_filename)
        with open(json_file, mode="r",
                  encoding=get_encoding(json_file)) as j:
            json_info = json.load(j)
            if 'imageHeight' in json_info and 'imageWidth' in json_info:
                h = json_info["imageHeight"]
                w = json_info["imageWidth"]

            node_size = xml_doc.createElement("size")
            node_width = xml_doc.createElement("width")
            node_width.appendChild(xml_doc.createTextNode(str(w)))
            node_size.appendChild(node_width)
            node_height = xml_doc.createElement("height")
            node_height.appendChild(xml_doc.createTextNode(str(h)))
            node_size.appendChild(node_height)
            node_depth = xml_doc.createElement("depth")
            node_depth.appendChild(xml_doc.createTextNode(str(3)))
            node_size.appendChild(node_depth)
            root.appendChild(node_size)
            for shape in json_info["shapes"]:
                if 'shape_type' in shape:
                    if shape["shape_type"] == "rectangle" or (shape['shape_type'] == 'mask' and len(shape['points']))==2:
                        (xmin, ymin), (xmax, ymax) = shape["points"]
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])
                    else:
                        continue
                else:
                    points = shape["points"]
                    points_num = len(points)
                    x = [points[i][0] for i in range(points_num)]
                    y = [points[i][1] for i in range(points_num)]
                    xmin = min(x)
                    xmax = max(x)
                    ymin = min(y)
                    ymax = max(y)
                label = shape["label"]
                node_obj = xml_doc.createElement("object")
                node_name = xml_doc.createElement("name")
                node_name.appendChild(xml_doc.createTextNode(label))
                node_obj.appendChild(node_name)
                node_diff = xml_doc.createElement("difficult")
                node_diff.appendChild(xml_doc.createTextNode(str(0)))
                node_obj.appendChild(node_diff)
                node_box = xml_doc.createElement("bndbox")
                node_xmin = xml_doc.createElement("xmin")
                node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                node_box.appendChild(node_xmin)
                node_ymin = xml_doc.createElement("ymin")
                node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                node_box.appendChild(node_ymin)
                node_xmax = xml_doc.createElement("xmax")
                node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                node_box.appendChild(node_xmax)
                node_ymax = xml_doc.createElement("ymax")
                node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                node_box.appendChild(node_ymax)
                node_obj.appendChild(node_box)
                root.appendChild(node_obj)
                if label not in name_list:
                    # name_list = list(name_list)
                    name_list.append(label)
        with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
            xml_doc.writexml(
                fxml,
                indent='\t',
                addindent='\t',
                newl='\n',
                encoding="utf-8")
        res.append(('Generating dataset from:' + json_file))


def VocGenerator(folder_data, value_data):
    res = []
    res_str = ""
    name_list = []
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
    dataset_type = 'labelme'
    try:
        assert os.path.exists(json_input_dir)
    except AssertionError as e:
        return 'The json folder does not exist!'
    try:
        assert os.path.exists(image_input_dir)
    except AssertionError as e:
        return 'The image folder does not exist!'
    try:
        assert abs(train_proportion + val_proportion \
                   + test_proportion - 1.0) < 1e-5
    except AssertionError as e:
        return 'The sum of pqoportion of training, validation and test datase must be 1!'
    new_image_dir = osp.join(output_dir, "images")
    if os.path.exists(new_image_dir):
        return "The directory is already exist, please remove the directory first"
    new_annotations_dir = osp.join(output_dir, "annotations")
    if os.path.exists(new_annotations_dir):
        return "The directory is already exist, please remove the directory first"

    convert(image_input_dir, json_input_dir, output_dir, name_list, res)
    with open(osp.join(output_dir, 'label_list' + '.txt'), 'w') as f:
        for i in name_list:
            f.write(str(i) + '\n')
    total_num = len(glob.glob(osp.join(json_input_dir, '*.json')))
    if train_proportion != 0:
        train_num = int(total_num * train_proportion)
    else:
        train_num = 0
    if val_proportion == 0.0:
        val_num = 0
    else:
        val_num = int(total_num * val_proportion)

    count = 1
    images_input = os.listdir(image_input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    json_file_names = {os.path.splitext(f)[0] for f in os.listdir(json_input_dir) if f.endswith('.json')}
    img_names = [
        f for f in images_input
        if os.path.splitext(f)[-1].lower() in image_extensions and os.path.splitext(f)[0] in json_file_names
    ]
    random.shuffle(img_names)
    trainset_output = osp.join(output_dir, 'train' + '.txt')
    valset_output = osp.join(output_dir, 'val.txt')
    testset_output = osp.join(output_dir, 'test.txt')
    trainset = open(trainset_output, 'w')
    valset = open(valset_output, 'w')
    testset = open(testset_output, 'w')
    for img_name in img_names:
        if count <= train_num:
            trainset.write(osp.join(output_dir + "/images", img_name) + ' ' +
                           osp.join(output_dir + "/annotations", os.path.splitext(img_name)[0] + '.xml') + '\n')
        else:
            if count <= train_num + val_num:
                valset.write(osp.join(output_dir + "/images", img_name) + ' ' +
                             osp.join(output_dir + "/annotations", os.path.splitext(img_name)[0] + '.xml') + '\n')
            else:
                testset.write(osp.join(output_dir + "/images", img_name) + ' ' +
                              osp.join(output_dir + "/annotations", os.path.splitext(img_name)[0] + '.xml') + '\n')
        count = count + 1
    for i in range(len(res)):
        res_str = res_str + '\n' + str(res[i])
    return res_str


# folder_data = ['C:/Users/fjl\Desktop/11\json', 'C:/Users/fjl\Desktop/11\image', 'C:/Users/fjl/Desktop/11/voc']
# value_data = [0.6, 0.2, 0.2]
# result = VocGenerator(folder_data, value_data)
# print(result)
# print(len(result))
