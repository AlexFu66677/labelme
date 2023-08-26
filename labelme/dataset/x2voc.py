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


class X2VOC(object):
    def __init__(self):
        pass

    def convert(self, image_dir, json_dir, dataset_save_dir,res):

        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "images")
        if osp.exists(new_image_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(new_image_dir))
        os.makedirs(new_image_dir)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                    osp.join(image_dir, img_name),
                    osp.join(new_image_dir, img_name))
        # Convert the json files.
        xml_dir = osp.join(dataset_save_dir, "annotations")
        if osp.exists(xml_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(xml_dir))
        os.makedirs(xml_dir)
        self.json2xml(new_image_dir, json_dir, xml_dir,res)


class LabelMe2VOC(X2VOC):
    def json2xml(self, image_dir, json_dir, xml_dir, res):
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
            with open(json_file, mode="r", \
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
                        if shape["shape_type"] != "rectangle":
                            continue
                        (xmin, ymin), (xmax, ymax) = shape["points"]
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])
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
            with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
                xml_doc.writexml(
                    fxml,
                    indent='\t',
                    addindent='\t',
                    newl='\n',
                    encoding="utf-8")
            res.append(('Generating dataset from:' + json_file))


def VocGenerator(folder_data,value_data):
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
    labelme2voc = LabelMe2VOC().convert
    labelme2voc(image_input_dir, json_input_dir, output_dir,res)
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
    img_names=[]
    for file in os.listdir(image_input_dir):
        if os.path.isfile(os.path.join(image_input_dir, file)):
            img_names.append(str(file))
    random.shuffle(img_names)
    trainset_output = osp.join(output_dir, 'train' + '.txt')
    valset_output = osp.join(output_dir, 'val.txt')
    testset_output = osp.join(output_dir, 'test.txt')
    trainset = open(trainset_output,'w')
    valset = open(valset_output, 'w')
    testset = open(testset_output, 'w')
    for img_name in img_names:
        if count <= train_num:
            trainset.write(osp.join(output_dir+"/images", img_name)+' ' +
                           osp.join(output_dir+"/annotations",os.path.splitext(img_name)[0]+'.xml')+'\n')
        else:
            if count <= train_num + val_num:
                valset.write(osp.join(output_dir+"/images", img_name)+' ' +
                             osp.join(output_dir + "/annotations", os.path.splitext(img_name)[0]+'.xml')+'\n')
            else:
                testset.write(osp.join(output_dir+"/images", img_name)+' ' +
                              osp.join(output_dir + "/annotations", os.path.splitext(img_name)[0]+'.xml')+'\n')
        count = count + 1
    for i in range(len(res)):
        res_str=res_str+'\n'+str(res[i])
    return res_str
# folder_data=['C:/Users/fjl\Desktop/11\json', 'C:/Users/fjl\Desktop/11\image', 'C:/Users/fjl/Desktop/11/voc']
# value_data=[0.4,0.3,0.3]
# result = VocGenerator(folder_data,value_data)
# print(result)
# print(len(result))
