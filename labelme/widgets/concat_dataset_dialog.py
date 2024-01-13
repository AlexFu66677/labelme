
from qtpy import QtWidgets
from .. import dataset
import os
import json
import shutil

class Concat_dataset(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset")

        self.folder_labels = []
        self.folder_inputs = []
        layout =QtWidgets.QVBoxLayout()
        # 添加下拉选择控件
        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("coco")

        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        folder_names = ["数据集1", "数据集2", "输出路径"]
        for i in range(3):
            folder_label = QtWidgets.QLabel(f"{folder_names[i]}:")
            folder_input = QtWidgets.QLineEdit()
            folder_input.setReadOnly(True)
            folder_button = QtWidgets.QPushButton("Select")
            folder_button.clicked.connect(lambda _, index=i: self.select_folder(index))

            self.folder_labels.append(folder_label)
            self.folder_inputs.append(folder_input)

            folder_layout = QtWidgets.QHBoxLayout()
            folder_layout.addWidget(folder_input)
            folder_layout.addWidget(folder_button)

            layout.addWidget(folder_label)
            layout.addLayout(folder_layout)


        self.result_label = QtWidgets.QLabel("Result:")
        self.result_text_edit = QtWidgets.QTextEdit()
        self.result_text_edit.setReadOnly(True)

        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text_edit)

        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(self.start)

        layout.addWidget(start_button)

        self.setLayout(layout)

    def load_json(self,filenamejson):
        with open(filenamejson) as f:
            raw_data = json.load(f)
        return raw_data

    def merge_train_coco_json(self,file_path1, file_path2, out):
        if os.path.exists(os.path.join(file_path1, 'annotations', 'instance_train.json')) and os.path.exists(
                os.path.join(file_path2, 'annotations', 'instance_train.json')):
            root_json_data = self.load_json(os.path.join(file_path1, 'annotations', 'instance_train.json'))
            raw_json_data = self.load_json(os.path.join(file_path2, 'annotations', 'instance_train.json'))
            temp_name = []
            temp_categories = []
            image_out_dir = os.path.join(out, 'train')
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)
            folder1 = os.path.join(file_path1, 'train')
            folder2 = os.path.join(file_path2, 'train')
            files1 = set(os.listdir(folder1))
            files2 = set(os.listdir(folder2))
            common_files = files1.intersection(files2)
            if common_files:
                raise ValueError(f"Error: Duplicate files found: {common_files}")
            # 将文件从第一个文件夹复制到输出文件夹
            for file in files1:
                src_path = os.path.join(folder1, file)
                dest_path = os.path.join(image_out_dir, file)
                shutil.copy(src_path, dest_path)
            # 将文件从第二个文件夹复制到输出文件夹
            for file in files2:
                src_path = os.path.join(folder2, file)
                dest_path = os.path.join(image_out_dir, file)
                shutil.copy(src_path, dest_path)

            for m in root_json_data["categories"]:
                if m['name'] not in temp_name:
                    temp_name.append(m['name'])
                    temp_categories.append(m)
            for n in raw_json_data["categories"]:
                if n['name'] not in temp_name:
                    temp_name.append(n['name'])
                    temp_categories.append(n)
            id_to_name = {category['id']: category['name'] for category in raw_json_data['categories']}
            for annotation in raw_json_data['annotations']:
                annotation['category_id'] = id_to_name[annotation['category_id']]

            for i in range(len(temp_categories)):
                temp_categories[i]['id'] = i + 1

            name_to_id = {category['name']: category['id'] for category in temp_categories}

            for annotation in raw_json_data['annotations']:
                annotation['category_id'] = name_to_id[annotation['category_id']]

            root_json_data['categories'] = temp_categories
            ###追加images
            root_images_len = len(root_json_data['images'])
            raw_images_len = len(raw_json_data['images'])
            for i in range(raw_images_len):
                raw_json_data['images'][i]['id'] = int(raw_json_data['images'][i]['id']) + int(root_images_len)
            root_json_data['images'].extend(raw_json_data['images'])
            root_annotations_len = len(root_json_data['annotations'])
            raw_annotations_len = len(raw_json_data['annotations'])
            for j in range(raw_annotations_len):
                raw_json_data['annotations'][j]['id'] = int(raw_json_data['annotations'][j]['id']) + int(
                    root_annotations_len)
                raw_json_data['annotations'][j]['image_id'] = int(raw_json_data['annotations'][j]['image_id']) + int(
                    root_images_len)
            root_json_data['annotations'].extend(raw_json_data['annotations'])
            json_out_dir = os.path.join(out, 'annotations')
            if not os.path.exists(json_out_dir):
                os.makedirs(json_out_dir)
            json_str = json.dumps(root_json_data)
            with open(os.path.join(json_out_dir, 'instance_train.json'), 'w') as json_file:
                json_file.write(json_str)
            return (f"训练集处理完成")
        else:
            return (f"缺少训练集文件")
    def merge_val_coco_json(self,file_path1, file_path2, out):
        if os.path.exists(os.path.join(file_path1, 'annotations', 'instance_val.json')) and os.path.exists(os.path.join(file_path2, 'annotations', 'instance_val.json')):
            root_json_data = self.load_json(os.path.join(file_path1, 'annotations', 'instance_val.json'))
            raw_json_data = self.load_json(os.path.join(file_path2, 'annotations', 'instance_val.json'))
            temp_name = []
            temp_categories = []
            image_out_dir = os.path.join(out, 'val')
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)
            folder1 = os.path.join(file_path1, 'val')
            folder2 = os.path.join(file_path2, 'val')
            files1 = set(os.listdir(folder1))
            files2 = set(os.listdir(folder2))
            common_files = files1.intersection(files2)
            if common_files:
                raise ValueError(f"Error: Duplicate files found: {common_files}")
            # 将文件从第一个文件夹复制到输出文件夹
            for file in files1:
                src_path = os.path.join(folder1, file)
                dest_path = os.path.join(image_out_dir, file)
                shutil.copy(src_path, dest_path)
            # 将文件从第二个文件夹复制到输出文件夹
            for file in files2:
                src_path = os.path.join(folder2, file)
                dest_path = os.path.join(image_out_dir, file)
                shutil.copy(src_path, dest_path)

            for m in root_json_data["categories"]:
                if m['name'] not in temp_name:
                    temp_name.append(m['name'])
                    temp_categories.append(m)
            for n in raw_json_data["categories"]:
                if n['name'] not in temp_name:
                    temp_name.append(n['name'])
                    temp_categories.append(n)
            id_to_name = {category['id']: category['name'] for category in raw_json_data['categories']}
            for annotation in raw_json_data['annotations']:
                annotation['category_id'] = id_to_name[annotation['category_id']]

            for i in range(len(temp_categories)):
                temp_categories[i]['id'] = i + 1

            name_to_id = {category['name']: category['id'] for category in temp_categories}

            for annotation in raw_json_data['annotations']:
                annotation['category_id'] = name_to_id[annotation['category_id']]

            root_json_data['categories'] = temp_categories
            ###追加images
            root_images_len = len(root_json_data['images'])
            raw_images_len = len(raw_json_data['images'])
            for i in range(raw_images_len):
                raw_json_data['images'][i]['id'] = int(raw_json_data['images'][i]['id']) + int(root_images_len)
            root_json_data['images'].extend(raw_json_data['images'])
            root_annotations_len = len(root_json_data['annotations'])
            raw_annotations_len = len(raw_json_data['annotations'])
            for j in range(raw_annotations_len):
                raw_json_data['annotations'][j]['id'] = int(raw_json_data['annotations'][j]['id']) + int(
                    root_annotations_len)
                raw_json_data['annotations'][j]['image_id'] = int(raw_json_data['annotations'][j]['image_id']) + int(
                    root_images_len)
            root_json_data['annotations'].extend(raw_json_data['annotations'])
            # root_data["categories"] = temp
            # print("共处理 {0} 个json文件".format(file_json_count))
            # print("共找到 {0} 个类别".format(str(root_json_data["categories"]).count('name', 0, len(str(root_data["categories"])))))
            json_out_dir = os.path.join(out, 'annotations')
            if not os.path.exists(json_out_dir):
              os.makedirs(json_out_dir)
            json_str = json.dumps(root_json_data)
            with open(os.path.join(json_out_dir, 'instance_val.json'), 'w') as json_file:
                  json_file.write(json_str)
            return (f"验证集处理完成")
        else:
            return (f"缺少验证集文件")

    def select_folder(self, index):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_inputs[index].setText(folder)
    def start(self):
        folder_data = [folder_input.text() for folder_input in self.folder_inputs]
        type_data = self.type_combobox.currentText()
        if type_data=='coco':
            result1 =self.merge_train_coco_json(folder_data[0], folder_data[1], folder_data[2])
            result2 =self.merge_val_coco_json(folder_data[0], folder_data[1], folder_data[2])
            result=result1+'\n'+result2
            self.result_text_edit.setText(result)
