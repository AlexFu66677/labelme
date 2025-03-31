import os
import numpy as np
from qtpy import QtWidgets, QtCore
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

class Dataset_analysis_Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Analysis Pro")
        self.setMinimumSize(1600, 1000)

        # 阈值配置
        self.size_thresholds = (100, 900)  # 小/中/大尺寸阈值（占图像面积比例）
        # 初始化数据结构
        self._init_data_structures()

        # 创建UI组件
        self._create_widgets()
        self._setup_layout()
        self._setup_connections()

        # 初始模式
        self.current_mode = "pixel_areas"

    def _init_data_structures(self):
        """初始化统计数据存储结构"""
        self.global_stats = {
            "total_images": 0,
            "total_classes": 0,
            "total_annotations": 0,
            "class_distribution": defaultdict(int),
            "max_area": 0,
            "min_area": float('inf')
        }

        self.class_stats = defaultdict(lambda: {
            "images": set(),  # 包含该类别的图像集合
            "annotations": 0,  # 总标注数
            "areas_ratios": [],  # 所有标注的面积比例
            "pixel_areas": [],  # 像素面积（需要实际图像尺寸）
            "aspect_ratios": []  # 宽高比（width/height）
        })

    def _create_widgets(self):
        """创建界面组件"""
        self.setStyleSheet("""
            QWidget {
                font-size: 11px;
                font-family: "Microsoft YaHei";  /* 设置字体 */
                font-weight: normal;  /* 设置字重 */
            }
        """)
        # 控制按钮
        self.load_btn = QtWidgets.QPushButton("📁 Load", self)
        self.mode_btns = {
            "areas_ratios": QtWidgets.QPushButton("📏 Areas Ratio", self),
            "pixel_areas": QtWidgets.QPushButton("🖼️ Label Pixel", self),
            "aspect_ratios": QtWidgets.QPushButton("📐 Aspect Ratio", self)
        }

        self.size_threshold_inputs = [
            QtWidgets.QLineEdit(str(self.size_thresholds[0])),
            QtWidgets.QLineEdit(str(self.size_thresholds[1]))
        ]
        self.apply_btn = QtWidgets.QPushButton("Apply Thresholds", self)
        # 统计表格
        self._create_tables()

        # 图表组件
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_bar.hide()

    def _create_tables(self):
        """创建三个统计表格"""
        # 顶部概览表
        self.summary_table = self._create_table(
            5, 2,
            ["Item","Value"],
            None
        )
        for row, text in enumerate(["Total Images", "Total Classes", "Total Annotations", "Max Area", "Min Area"]):
            self.summary_table.setItem(row, 0, self._create_table_item(text))
        # 中间类别概览表
        self.class_overview_table = self._create_table(
            0, 5,
            ["Class", "Images", "Images(%)", "Annotations", "Annotations(%)"],
            None
        )

        # 底部详细分析表
        self.detail_table = self._create_table(
            0, 8,  # 增加三列
            ["Class", "Small", "S(%)","Medium","M(%)", "Large","L(%)", "Total"],  # 增加百分比列
            None
        )

        # 设置表格样式
        for table in [self.summary_table, self.class_overview_table, self.detail_table]:
            table.setStyleSheet("""
                QTableWidget { 
                    font-size: 11px; 
                    selection-background-color: #e0f0f; 
                }
                QTableWidget::item { 
                    padding: 3px; 
                }
            """)

    def _create_table(self, rows, cols, h_headers, v_headers):
        """通用表格创建方法"""
        table = QtWidgets.QTableWidget(rows, cols)
        table.setHorizontalHeaderLabels(h_headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # 增强表头样式
        table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                border: 1px solid #ccc;
                padding: 5px;
            }
        """)

        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        return table

    def _setup_layout(self):
        """设置界面布局"""
        main_layout = QtWidgets.QVBoxLayout(self)

        # 顶部控制栏
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(self.load_btn)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        # 主内容区域
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # 左侧统计面板
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 顶部概览组
        summary_group = QtWidgets.QGroupBox("Global Overview")
        summary_group.setLayout(QtWidgets.QVBoxLayout())
        summary_group.layout().addWidget(self.summary_table)
        left_layout.addWidget(summary_group)

        # 中间类别概览组
        mid_group = QtWidgets.QGroupBox("Class Overview")
        mid_group.setLayout(QtWidgets.QVBoxLayout())
        mid_group.layout().addWidget(self.class_overview_table)
        left_layout.addWidget(mid_group)

        # 按钮和阈值设置区域
        btn_threshold_group = QtWidgets.QGroupBox("Controls")
        btn_threshold_layout = QtWidgets.QVBoxLayout()

        # 添加模式按钮
        btn_layout = QtWidgets.QHBoxLayout()
        for btn in self.mode_btns.values():
            btn.setCheckable(True)
            btn_layout.addWidget(btn)
        self.mode_btns["pixel_areas"].setChecked(True)
        btn_threshold_layout.addLayout(btn_layout)

        # 添加阈值输入
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("Thresholds:"))
        for input in self.size_threshold_inputs:
            input.setFixedWidth(80)
            threshold_layout.addWidget(input)
        threshold_layout.addWidget(self.apply_btn)
        btn_threshold_layout.addLayout(threshold_layout)

        btn_threshold_group.setLayout(btn_threshold_layout)
        left_layout.addWidget(btn_threshold_group)

        # 底部详细分析组
        bottom_group = QtWidgets.QGroupBox("Detailed Analysis")
        bottom_group.setLayout(QtWidgets.QVBoxLayout())
        bottom_group.layout().addWidget(self.detail_table)
        left_layout.addWidget(bottom_group)

        # 右侧图表区
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QGridLayout(right_panel)

        # 创建四个图表
        self.figure1 = plt.figure(figsize=(6, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        right_layout.addWidget(self.canvas1, 0, 0)  # 左上

        self.figure2 = plt.figure(figsize=(6, 4))
        self.canvas2 = FigureCanvas(self.figure2)
        right_layout.addWidget(self.canvas2, 1, 0)  # 左下

        self.figure3 = plt.figure(figsize=(6, 4))
        self.canvas3 = FigureCanvas(self.figure3)
        right_layout.addWidget(self.canvas3, 0, 1)  # 右上

        self.figure4 = plt.figure(figsize=(6, 4))
        self.canvas4 = FigureCanvas(self.figure4)
        right_layout.addWidget(self.canvas4, 1, 1)  # 右下

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([500, 1000])

        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(main_splitter)
    def _setup_connections(self):
        """连接信号与槽"""
        self.load_btn.clicked.connect(self.load_dataset)
        for mode, btn in self.mode_btns.items():
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))

        # 添加阈值应用按钮的连接
        self.apply_btn.clicked.connect(self._apply_thresholds)

    def _apply_thresholds(self):
        """应用新的阈值"""
        try:
            new_thresholds = (
                float(self.size_threshold_inputs[0].text()),
                float(self.size_threshold_inputs[1].text())
            )
            if new_thresholds[0] >= new_thresholds[1]:
                raise ValueError("第一个阈值必须小于第二个阈值")
            self.size_thresholds = new_thresholds
            self.update_detail_table()
            self.update_visualization()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", f"无效的阈值输入: {str(e)}")

    def switch_mode(self, mode):
        """切换分析模式"""
        self.current_mode = mode
        for btn in self.mode_btns.values():
            btn.setChecked(False)
        self.mode_btns[mode].setChecked(True)

        self.update_detail_table()
        self.update_visualization()

    def load_dataset(self):
        """加载数据集"""
        dataset_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select YOLO Dataset")
        if not dataset_dir:
            return

        # 重置数据
        self._init_data_structures()

        try:
            # 获取图像和标签文件
            all_files = os.listdir(dataset_dir)
            image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            label_files = [f for f in all_files if f.endswith('.txt')]

            self.global_stats["total_images"] = len(image_files)
            image_basenames = {os.path.splitext(f)[0] for f in image_files}

            self.progress_bar.setRange(0, len(label_files))
            self.progress_bar.show()

            # 使用线程池处理每个标签文件
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_label_file, dataset_dir,image_files, label_file, image_basenames): label_file for
                    label_file in label_files}
                for future in as_completed(futures):
                    idx = list(futures.values()).index(futures[future])
                    try:
                        future.result()  # 捕获异常
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "Error", f"处理文件 {futures[future]} 失败：{str(e)}")
                    self.progress_bar.setValue(idx + 1)
                    QtWidgets.QApplication.processEvents()

            # 计算全局类别数
            self.global_stats["total_classes"] = len(self.class_stats)

            # 更新界面
            self.update_tables()
            self.update_visualization()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"加载数据集失败：{str(e)}")
        finally:
            self.progress_bar.hide()

    def process_label_file(self, dataset_dir,image_files, label_file, image_basenames):
        base_name = os.path.splitext(label_file)[0]
        if base_name not in image_basenames:
            return

        image_path = os.path.join(dataset_dir, base_name + os.path.splitext(image_files[0])[1])
        img = plt.imread(image_path)
        img_height, img_width = img.shape[:2]
        label_path = os.path.join(dataset_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # 解析标注数据
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # 转换为绝对尺寸
                abs_width = width * img_width
                abs_height = height * img_height
                pixel_area = abs_width * abs_height

                # 更新全局统计
                self.global_stats["total_annotations"] += 1
                self.global_stats["class_distribution"][class_id] += 1
                self.global_stats["max_area"] = max(self.global_stats["max_area"], pixel_area)
                self.global_stats["min_area"] = min(self.global_stats["min_area"], pixel_area)

                # 更新类别统计
                self.class_stats[class_id]["images"].add(base_name)
                self.class_stats[class_id]["annotations"] += 1
                self.class_stats[class_id]["pixel_areas"].append(abs_width * abs_height)
                self.class_stats[class_id]["areas_ratios"].append(width * height)
                self.class_stats[class_id]["aspect_ratios"].append(width / height if height != 0 else 0)

    def update_tables(self):
        """更新所有表格数据"""
        # 更新顶部概览表

        self.summary_table.setItem(0, 1, self._create_table_item(str(self.global_stats["total_images"])))
        self.summary_table.setItem(1, 1, self._create_table_item(str(self.global_stats["total_classes"])))
        self.summary_table.setItem(2, 1, self._create_table_item(str(self.global_stats["total_annotations"])))
        self.summary_table.setItem(3, 1, self._create_table_item(f"{self.global_stats['max_area']:.2f}"))
        self.summary_table.setItem(4, 1, self._create_table_item(f"{self.global_stats['min_area']:.2f}"))

        # 更新类别概览表
        total_images = self.global_stats["total_images"]
        total_annotations = self.global_stats["total_annotations"]
        self.class_overview_table.setRowCount(len(self.class_stats))

        for row, class_id in enumerate(sorted(self.class_stats.keys())):
            stats = self.class_stats[class_id]
            img_count = len(stats["images"])
            img_percent = img_count / total_images if total_images else 0
            anno_count = stats["annotations"]
            anno_percent = anno_count / total_annotations if total_annotations else 0

            items = [
                str(class_id),
                str(img_count),
                f"{img_percent:.1%}",
                str(anno_count),
                f"{anno_percent:.1%}"
            ]

            for col, text in enumerate(items):
                self.class_overview_table.setItem(row, col, self._create_table_item(text))

        # 更新详细分析表
        self.update_detail_table()

    def update_detail_table(self):
        """更新详细分析表（根据当前模式）"""
        self.detail_table.setRowCount(len(self.class_stats))

        for row, class_id in enumerate(sorted(self.class_stats.keys())):
            stats = self.class_stats[class_id]

            if self.current_mode == "areas_ratios":
                values = self._calculate_areas_ratios_distribution(stats["areas_ratios"])
            elif self.current_mode == "pixel_areas":
                values = self._calculate_pixel_distribution(stats["pixel_areas"])
            elif self.current_mode == "aspect_ratios":
                values = self._calculate_aspect_distribution(stats["aspect_ratios"])

            total = sum(values)
            items = [str(class_id)]

            # 将数量和百分比相邻显示
            for v in values:
                items.append(f"{v}" if total else "N/A")
                if total:
                    items.append(f"{(v / total) * 100:.1f}%")
                else:
                    items.append("N/A")

            items.append(str(total))

            for col, text in enumerate(items):
                self.detail_table.setItem(row, col, self._create_table_item(text))

    def _calculate_areas_ratios_distribution(self, areas):
        """计算尺寸分布"""
        return (
            sum(1 for a in areas if a < self.size_thresholds[0]),
            sum(1 for a in areas if self.size_thresholds[0] <= a < self.size_thresholds[1]),
            sum(1 for a in areas if a >= self.size_thresholds[1])
        )

    def _calculate_pixel_distribution(self, pixel_areas):
        return (
            sum(1 for a in pixel_areas if a < self.size_thresholds[0]),
            sum(1 for a in pixel_areas if self.size_thresholds[0] <= a < self.size_thresholds[1]),
            sum(1 for a in pixel_areas if a >= self.size_thresholds[1])
        )

    def _calculate_aspect_distribution(self, aspect_ratios):
        """计算宽高比分布"""
        return (
            sum(1 for r in aspect_ratios if r < self.size_thresholds[0]),
            sum(1 for r in aspect_ratios if self.size_thresholds[0] <= r < self.size_thresholds[1]),
            sum(1 for r in aspect_ratios if r >= self.size_thresholds[1])
        )

    def update_visualization(self):
        """更新可视化图表"""
        self.figure1.clear()
        self.figure2.clear()
        self.figure3.clear()
        self.figure4.clear()
        # 第一个图表：不区分类别的3种大小分布
        self._plot_overall_size_distribution(self.figure1)

        # 第二个图表：不区分类别的10类标注大小分布
        self._plot_unclassified_size_distribution(self.figure2)

        # 第三个图表：区分类别的10类标注大小分布
        self._plot_classified_size_distribution(self.figure3)

        # 第四个图表：区分类别的3种大小分布
        if self.current_mode == "areas_ratios":
            self._plot_size_distribution(self.figure4)
        elif self.current_mode == "aspect_ratios":
            self._plot_aspect_distribution(self.figure4)
        elif self.current_mode == "pixel_areas":
            self._plot_pixel_coverage(self.figure4)

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

    def _plot_overall_size_distribution(self, figure):
        """绘制不区分类别所有标注框大中小三类的数量"""
        ax = figure.add_subplot(111)
        if self.current_mode == "areas_ratios":
            all_pixel_areas = [area for stats in self.class_stats.values() for area in stats["areas_ratios"]]
            small = sum(1 for a in all_pixel_areas if a < self.size_thresholds[0])
            medium = sum(1 for a in all_pixel_areas if self.size_thresholds[0] <= a < self.size_thresholds[1])
            large = sum(1 for a in all_pixel_areas if a >= self.size_thresholds[1])
        elif self.current_mode == "aspect_ratios":
            all_pixel_areas = [area for stats in self.class_stats.values() for area in stats["aspect_ratios"]]
            small = sum(1 for a in all_pixel_areas if a < self.size_thresholds[0])
            medium = sum(1 for a in all_pixel_areas if self.size_thresholds[0] <= a < self.size_thresholds[1])
            large = sum(1 for a in all_pixel_areas if a >= self.size_thresholds[1])
        elif self.current_mode == "pixel_areas":
            all_pixel_areas = [area for stats in self.class_stats.values() for area in stats["pixel_areas"]]
            small = sum(1 for a in all_pixel_areas if a < self.size_thresholds[0])
            medium = sum(1 for a in all_pixel_areas if self.size_thresholds[0] <= a < self.size_thresholds[1])
            large = sum(1 for a in all_pixel_areas if a >= self.size_thresholds[1])
        # 计算所有标注的分布
        labels = ["Small", "Medium", "Large"]
        sizes = [small, medium, large]
        colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

        def autopct_format(pct, all_vals):
            absolute = int(round(pct / 100. * sum(all_vals)))  # 计算具体数量
            return f"{absolute}\n({pct:.1f}%)" if absolute > 0 else ""  # 避免零值显示

        ax.pie(sizes, labels=labels, colors=colors, autopct=lambda pct: autopct_format(pct, sizes),
               startangle=140, wedgeprops={'edgecolor': 'black'})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        figure.tight_layout()

    def _plot_unclassified_size_distribution(self, figure):
        """绘制不区分类别的标注大小分布"""
        ax = figure.add_subplot(111)

        # 计算所有标注的分布
        all_pixel_areas = [area for stats in self.class_stats.values() for area in stats["pixel_areas"]]
        bins = [i * 100 for i in range(12)]  # 0, 100, 200, ..., 1000
        counts, bin_edges = np.histogram(all_pixel_areas, bins=bins)
        counts[-1] = len(all_pixel_areas) - sum(counts[:-1])  # 最后一个区间包含大于等于1000的像素面积
        bars = ax.bar(bin_edges[:-1], counts, width=100, edgecolor='black', align='edge')
        ax.bar_label(bars, labels=[str(c) for c in counts], padding=3)
        ax.set_xlabel("Pixel Area")
        ax.set_ylabel("counts")
        ax.set_title("All class size distribution")
        ax.set_xticks(bin_edges[:-1])  # 设置 x 轴刻度为每个区间的起始点
        ax.set_xticklabels([f"{b}" if b < 1100 else f">{b}" for b in bin_edges[:-1]], rotation=45)

        figure.tight_layout()

    def _plot_classified_size_distribution(self, figure):
        """绘制区分类别的标注大小分布"""
        ax = figure.add_subplot(111)
        class_ids = sorted(self.class_stats.keys())
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, np.inf]
        bin_labels = [f"{bins[i]}-{bins[i + 1]}" if bins[i + 1] != np.inf else f">{bins[i]}" for i in
                      range(len(bins) - 1)]
        colors = sns.color_palette("rainbow", len(bin_labels), desat=0.5)
        # colors = plt.cm.Pastel1(np.linspace(0, 1,len(bin_labels) ))  # 颜色方案
        class_bin_counts = {cid: np.histogram(self.class_stats[cid]["pixel_areas"], bins=bins)[0] for cid in class_ids}

        # 绘制柱状图（堆叠）
        bottom = np.zeros(len(class_ids))  # 记录当前类别的底部高度
        bar_width = 0.6  # 柱子宽度

        for i, (label, color) in enumerate(zip(bin_labels, colors)):
            values = [class_bin_counts[cid][i] for cid in class_ids]
            bars = ax.bar(class_ids, values, width=bar_width, bottom=bottom, color=color, edgecolor='black',
                          label=label)
            # ax.bar_label(bars, labels=[str(v) if v > 0 else "" for v in values], padding=3, fontsize=8)
            bottom += values  # 更新底部高度

        # 设置 X 轴
        ax.set_xticks(class_ids)
        ax.set_xticklabels(class_ids, rotation=45)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Counts")
        ax.set_title("Class-wise Size Distribution")

        # 添加图例
        ax.legend(title="Pixel Area", bbox_to_anchor=(1.05, 1), loc='upper left')

        figure.tight_layout()

    def _plot_size_distribution(self, figure):
        """绘制尺寸分布图"""
        ax = figure.add_subplot(111)
        class_ids = sorted(self.class_stats.keys())

        # 准备数据
        small = []
        medium = []
        large = []
        for cid in class_ids:
            dist = self._calculate_areas_ratios_distribution(self.class_stats[cid]["areas_ratios"])
            small.append(dist[0])
            medium.append(dist[1])
            large.append(dist[2])

        # 堆叠柱状图
        bar_width = 0.6
        x = np.arange(len(class_ids))
        ax.bar(x, small, bar_width, label="Small", color="#66c2a5")
        ax.bar(x, medium, bar_width, bottom=small, label="Medium", color="#fc8d62")
        ax.bar(x, large, bar_width, bottom=np.array(small) + np.array(medium), label="Large", color="#8da0cb")

        ax.set_xticks(x)
        ax.set_xticklabels(class_ids)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Annotation Count")
        ax.set_title("Size Distribution per Class")
        ax.legend()

        self.figure.tight_layout()

    def _plot_aspect_distribution(self, figure):
        """绘制宽高比分布图"""
        ax = figure.add_subplot(111)
        class_ids = sorted(self.class_stats.keys())

        # 准备数据
        small = []
        medium = []
        large = []
        for cid in class_ids:
            dist = self._calculate_pixel_distribution(self.class_stats[cid]["aspect_ratios"])
            small.append(dist[0])
            medium.append(dist[1])
            large.append(dist[2])

        # 堆叠柱状图
        bar_width = 0.6
        x = np.arange(len(class_ids))
        ax.bar(x, small, bar_width, label="Small", color="#66c2a5")
        ax.bar(x, medium, bar_width, bottom=small, label="Medium", color="#fc8d62")
        ax.bar(x, large, bar_width, bottom=np.array(small) + np.array(medium), label="Large", color="#8da0cb")

        ax.set_xticks(x)
        ax.set_xticklabels(class_ids)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Annotation Count")
        ax.set_title("Size Distribution per Class")
        ax.legend()

        self.figure.tight_layout()

    def _plot_pixel_coverage(self, figure):
        """绘制尺寸分布图"""
        ax = figure.add_subplot(111)
        class_ids = sorted(self.class_stats.keys())

        # 准备数据
        small = []
        medium = []
        large = []
        for cid in class_ids:
            dist = self._calculate_pixel_distribution(self.class_stats[cid]["pixel_areas"])
            small.append(dist[0])
            medium.append(dist[1])
            large.append(dist[2])

        # 堆叠柱状图
        bar_width = 0.6
        x = np.arange(len(class_ids))
        ax.bar(x, small, bar_width, label="Small", color="#66c2a5")
        ax.bar(x, medium, bar_width, bottom=small, label="Medium", color="#fc8d62")
        ax.bar(x, large, bar_width, bottom=np.array(small) + np.array(medium), label="Large", color="#8da0cb")

        ax.set_xticks(x)
        ax.set_xticklabels(class_ids)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Annotation Count")
        ax.set_title("Size Distribution per Class")
        ax.legend()

        self.figure.tight_layout()

    def _create_table_item(self, text):
        """创建带居中对齐的表格项"""
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        return item


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Dataset_analysis_Dialog()
    window.show()
    sys.exit(app.exec_())
