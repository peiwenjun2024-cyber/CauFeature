import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tensorflow as tf
import logging

# 设置TensorFlow日志级别为WARNING（屏蔽INFO及以下级别的日志）
tf.get_logger().setLevel(logging.ERROR)
import os
import time

import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
import shared_globals

# 允许动态增长内存，避免一次性占满
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from model_train_v6 import train, save_model
from feature_perturbation_v7_english import CausalFeaturePerturbation
from causal_graph_build_v19_english import build_causal_graph, Feature
from redundantRecover_v10_english import PathShapleyModule

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FeatureSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CauFeature: A Causal-Aware Framework for Robust and Interpretable Feature Selection")
        self.root.geometry("1000x700")  # 适当扩大窗口

        # 存储所有步骤的图表数据
        self.plot_history = []
        self.current_plot_index = -1


        # ---------- 1. 步骤说明标签 ----------
        step_text = """1. First, open the tool and select the dataset and model (we'll use CNN today);
2. Click "Start Feature Selection" — you'll see it first run adaptive perturbation;
3. Next, it will plot the feature contributions;
4. Then the causal graph pops up — Only causal relationships are retained;
5. Finally, it will add collaborative features and output the final feature set. Done!"""
        self.step_label = tk.Label(
            root,
            text=step_text,
            justify=tk.LEFT,
            wraplength=900  # 扩大换行宽度
        )
        self.step_label.pack(pady=10)

        # 数据集文件夹路径
        self.dataset_dir = "/root/tmp/causalFeatureX_LSTM_1003/dataset1"

        # ---------- 0. 数据集和模型选择区域 ----------
        selection_frame = tk.Frame(root)
        selection_frame.pack(pady=10, fill=tk.X, padx=20)

        # 数据集选择（从文件夹读取）
        tk.Label(selection_frame, text="Select Dataset:").pack(side=tk.LEFT, padx=(0, 10))
        self.dataset_var = tk.StringVar()
        self.dataset_options = self.get_dataset_list()  # 调用类的成员方法
        self.dataset_combobox = ttk.Combobox(
            selection_frame,
            textvariable=self.dataset_var,
            values=self.dataset_options,
            state="readonly",
            width=25
        )
        if self.dataset_options:
            self.dataset_var.set(self.dataset_options[0])
        self.dataset_combobox.pack(side=tk.LEFT, padx=(0, 20))

        # 模型选择
        tk.Label(selection_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))
        self.model_var = tk.StringVar(value="CNN")
        model_options = ["CNN"]
        self.model_combobox = ttk.Combobox(
            selection_frame,
            textvariable=self.model_var,
            values=model_options,
            state="readonly",
            width=20
        )
        self.model_combobox.pack(side=tk.LEFT)

        shared_globals.init()
        # 文件路径处理（兼容不同操作系统）
        self.data_path = os.path.join(self.dataset_dir, self.dataset_var.get())
        self.model_name = self.model_var.get()

        # ---------- 2. 布局调整：Loaded、Start按钮、Step文本 ----------
        layout_frame = tk.Frame(root)
        layout_frame.pack(pady=5, fill=tk.X, padx=20)

        # Loaded 提示（左侧）
        self.status_var = tk.StringVar(value="Loaded 1 dataset")
        self.status_label = tk.Label(
            layout_frame,
            textvariable=self.status_var,
            font=("Arial", 9),
            fg="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 20))

        # Start Feature Selection 按钮（中间）
        self.start_btn = tk.Button(
            layout_frame,
            text="Start Feature Selection",
            command=self.start_feature_selection,
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white"
        )
        # 若没有数据集，禁用按钮
        if not self.dataset_options:
            self.start_btn.config(state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 20))

        # Step 文本（右侧）
        self.step_text_var = tk.StringVar(value="Step 0/5: Ready")
        self.step_text_label = tk.Label(
            layout_frame,
            textvariable=self.step_text_var,
            font=("Arial", 9),
            fg="#2196F3"
        )
        self.step_text_label.pack(side=tk.LEFT)

        # ---------- 3. 进度条 ----------
        self.progress = ttk.Progressbar(
            root,
            orient=tk.HORIZONTAL,
            length=500,
            mode='determinate'
        )
        self.progress.pack(pady=5, fill=tk.X, padx=20)

        # 数据集状态提示（保留原功能，可根据需要调整显示）
        self.dataset_status = tk.Label(root, text="", font=("Arial", 9), fg="red")
        self.dataset_status.pack(pady=2)
        self.update_dataset_status()

        # ---------- 4. 图表导航按钮（带提示文字） ----------
        nav_frame = tk.Frame(root)
        nav_frame.pack(pady=5)

        self.prev_btn = tk.Button(
            nav_frame,
            text="← Previous",
            command=self.show_previous_plot,
            state=tk.DISABLED
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        # Previous 按钮提示
        self.prev_hint = tk.Label(
            nav_frame,
            font=("Arial", 8),
            fg="gray"
        )
        self.prev_hint.pack(side=tk.LEFT)

        self.next_btn = tk.Button(
            nav_frame,
            text="Next →",
            command=self.show_next_plot,
            state=tk.DISABLED
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        # Next 按钮提示
        self.next_hint = tk.Label(
            nav_frame,
            text="(After CauFeature is completed, you can look left and right.)",
            font=("Arial", 8),
            fg="gray"
        )
        self.next_hint.pack(side=tk.LEFT)

        # ---------- 5. 绘图框架 ----------
        self.frame = tk.Frame(root, bd=1, relief=tk.SUNKEN)  # 加边框区分绘图区
        self.frame.pack(pady=5, fill=tk.BOTH, expand=True, padx=20)

        # 实例变量初始化（避免未定义错误）
        self.original_feature_len = 0  # 原始特征维度（绑定到实例）
        self.feature_perturb_time = 0  # 特征扰动耗时
        self.causal_graph_time = 0  # 因果图耗时
        self.feature_recover_time = 0  # 冗余恢复耗时

    def get_dataset_list(self):
        """获取数据集文件夹中的所有非隐藏文件/文件夹"""
        try:
            if not os.path.exists(self.dataset_dir):
                return []
            # 获取文件夹内容并过滤隐藏文件
            items = os.listdir(self.dataset_dir)
            filtered_items = [item for item in items if not item.startswith('.')]
            return filtered_items
        except Exception as e:
            print(f"读取数据集失败: {str(e)}")
            return []

    def update_dataset_status(self):
        """更新数据集路径状态提示"""
        if not os.path.exists(self.dataset_dir):
            self.status_var.set(f"Path does not exist: {self.dataset_dir}")
            self.status_label.config(fg="red")
        elif not self.dataset_options:
            self.status_var.set(f"Folder is empty: {self.dataset_dir}")
            self.status_label.config(fg="red")
        else:
            self.status_var.set(f"Loaded {len(self.dataset_options)} dataset")
            self.status_label.config(fg="green")

    def start_feature_selection(self):
        """执行特征选择流程"""
        # 清空历史图表
        self.plot_history = []
        self.current_plot_index = -1
        self.update_nav_buttons()

        selected_dataset = self.dataset_var.get()
        selected_model = self.model_var.get()
        self.data_path = os.path.join(self.dataset_dir, selected_dataset)
        self.model_name = selected_model  # 修正模型名赋值

        print(f"Selected Dataset: {selected_dataset} (Path: {self.data_path})")
        print(f"Selected Model: {selected_model}")

        # 禁用控件防止重复点击
        self.start_btn.config(state=tk.DISABLED)
        self.dataset_combobox.config(state="disabled")
        self.model_combobox.config(state="disabled")

        self.step_text_var.set("Step 1/5: Initial Model Training...")


        # 初始模型训练（进度条：0%→20%）
        self.progress["value"] = 0
        self.root.update_idletasks()
        try:
            shared_globals.model = train(self.data_path, shared_globals.filtered_features)
            save_model(shared_globals.model, self.data_path, self.model_name)
            self.show_origin_model()
            # 记录原始特征维度（绑定到实例self，供后续使用）
            self.original_feature_len = shared_globals.feature_data.shape[1]
            self.progress["value"] = 20
            self.root.update_idletasks()
            time.sleep(1)
        except Exception as e:
            self.step_text_var.set(f"Initial Training Failed: {str(e)}")
            messagebox.showerror("Error", f"Initial Training Failed: {str(e)}")
            self.start_btn.config(state=tk.NORMAL)
            self.dataset_combobox.config(state="readonly")
            self.model_combobox.config(state="readonly")
            return

        # 执行特征贡献度计算（进度条：20%→40%）

        self.start_feature_contribution()
        self.start_causal_graph()
        self.start_feature_supply()


        shared_globals.model = None
        train(self.data_path, shared_globals.filtered_features)
        self.progress["value"] = 100
        self.root.update_idletasks()
        self.show_after_model()
        self.show_result()

    def show_origin_model(self):
        """显示模型训练的分类报告（控制仅添加一次到历史记录）"""
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 从全局变量获取分类报告（字典格式）
        report_dict = getattr(shared_globals, 'originalreport', None)

        # 处理报告不存在的情况
        if not report_dict:
            tk.Label(
                self.frame,
                text="No classification report available.\nPlease train the model first.",
                font=("Arial", 10),
                fg="red",
                justify=tk.CENTER,
                wraplength=800
            ).pack(expand=True, pady=50)
            self.step_text_var.set("No Classification Report Found")
            return

        # 解析字典为表格行数据（省略，同原代码）
        headers = ["class", "precision", "recall", "f1-score", "support"]
        data = []
        # （处理各类别行和汇总行的代码不变）
        for class_label, metrics in report_dict.items():
            if class_label not in ["accuracy", "macro avg", "weighted avg"]:
                data.append([
                    str(class_label),
                    f"{metrics['precision']:.5f}",
                    f"{metrics['recall']:.5f}",
                    f"{metrics['f1-score']:.5f}",
                    f"{metrics['support']:.0f}"
                ])
        for summary_type in ["accuracy", "macro avg", "weighted avg"]:
            if summary_type in report_dict:
                metrics = report_dict[summary_type]
                row = [summary_type]
                if summary_type == "accuracy":
                    row.extend(["-", "-", f"{metrics:.5f}", report_dict["macro avg"]["support"]])
                else:
                    row.extend([
                        f"{metrics['precision']:.5f}",
                        f"{metrics['recall']:.5f}",
                        f"{metrics['f1-score']:.5f}",
                        f"{metrics['support']:.0f}"
                    ])
                data.append(row)

        # 创建表格和UI（省略，同原代码）
        report_frame = tk.Frame(self.frame, bd=2, relief=tk.GROOVE, padx=15, pady=15)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        title_text = "Original Model Classification Report"
        tk.Label(report_frame, text=title_text, font=("Arial", 12, "bold"), fg="#2C3E50", pady=10).pack(anchor=tk.W)
        tree_frame = tk.Frame(report_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(tree_frame, columns=headers, show="headings")
        for header in headers:
            tree.heading(header, text=header)
            tree.column(header, width=120 if header in ["precision", "recall", "f1-score"] else 100, anchor="center")
        for row_data in data:
            tree.insert("", tk.END, values=row_data)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        plot_name = 'Original Model Classification Report'

        # --------------------------
        # 关键修改：检查是否已存在，仅首次添加
        # --------------------------
        # 检查历史记录中是否已有该报告
        exists = any(name == plot_name for name, _ in self.plot_history)
        if not exists:  # 仅在不存在时添加
            self.plot_history.append((plot_name, None))
        # 定位当前报告在历史记录中的索引（取首次出现的位置）
        self.current_plot_index = next(
            i for i, (name, _) in enumerate(self.plot_history)
            if name == plot_name
        )
        self.update_nav_buttons()


        self.step_text_var.set("Step 2/5: Feature Perturbation...")
        self.frame.update_idletasks()

    def show_after_model(self):
        """显示模型训练的分类报告（控制仅添加一次到历史记录）"""
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 从全局变量获取分类报告（字典格式）
        report_dict = getattr(shared_globals, 'afterreport', None)

        # 处理报告不存在的情况
        if not report_dict:
            tk.Label(
                self.frame,
                text="No classification report available.\nPlease train the model first.",
                font=("Arial", 10),
                fg="red",
                justify=tk.CENTER,
                wraplength=800
            ).pack(expand=True, pady=50)
            self.step_text_var.set("No Classification Report Found")
            return

        # 解析字典为表格行数据（省略，同原代码）
        headers = ["class", "precision", "recall", "f1-score", "support"]
        data = []
        # （处理各类别行和汇总行的代码不变）
        for class_label, metrics in report_dict.items():
            if class_label not in ["accuracy", "macro avg", "weighted avg"]:
                data.append([
                    str(class_label),
                    f"{metrics['precision']:.5f}",
                    f"{metrics['recall']:.5f}",
                    f"{metrics['f1-score']:.5f}",
                    f"{metrics['support']:.0f}"
                ])
        for summary_type in ["accuracy", "macro avg", "weighted avg"]:
            if summary_type in report_dict:
                metrics = report_dict[summary_type]
                row = [summary_type]
                if summary_type == "accuracy":
                    row.extend(["-", "-", f"{metrics:.5f}", report_dict["macro avg"]["support"]])
                else:
                    row.extend([
                        f"{metrics['precision']:.5f}",
                        f"{metrics['recall']:.5f}",
                        f"{metrics['f1-score']:.5f}",
                        f"{metrics['support']:.0f}"
                    ])
                data.append(row)

        # 创建表格和UI（省略，同原代码）
        report_frame = tk.Frame(self.frame, bd=2, relief=tk.GROOVE, padx=15, pady=15)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        title_text = "CauFeature Selected Model Classification Report"
        tk.Label(report_frame, text=title_text, font=("Arial", 12, "bold"), fg="#2C3E50", pady=10).pack(anchor=tk.W)
        tree_frame = tk.Frame(report_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(tree_frame, columns=headers, show="headings")
        for header in headers:
            tree.heading(header, text=header)
            tree.column(header, width=120 if header in ["precision", "recall", "f1-score"] else 100,
                        anchor="center")
        for row_data in data:
            tree.insert("", tk.END, values=row_data)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        plot_name = 'CauFeature Selected Model Classification Report'

        # --------------------------
        # 关键修改：检查是否已存在，仅首次添加
        # --------------------------
        exists = any(name == plot_name for name, _ in self.plot_history)
        if not exists:  # 仅在不存在时添加
            self.plot_history.append((plot_name, None))
        # 定位当前报告在历史记录中的索引
        self.current_plot_index = next(
            i for i, (name, _) in enumerate(self.plot_history)
            if name == plot_name
        )
        self.update_nav_buttons()

        self.frame.update_idletasks()



    def start_feature_contribution(self):
        """特征扰动计算（含耗时统计）"""


        start_time = time.time()
        try:
            # 初始化特征扰动实例
            perturb = CausalFeaturePerturbation(shared_globals.model, shared_globals.data)
            perturb.run_perturbation()
            perturb.initialize_original_preds(shared_globals.data, shared_globals.model)
            # 按影响度降序排序
            shared_globals.feature_max_abs.sort(key=lambda x: x[1], reverse=True)
            # 打印特征贡献度（控制台调试用）
            print("\nFeature Contributions (Max JS Divergence, Sorted):")
            for idx, (feat_name, contrib) in enumerate(shared_globals.feature_max_abs[:10]):  # 只打印前10个
                print(f"  Top {idx + 1}: {feat_name} → {round(contrib, 4)}")
            print("=" * 60)
        except Exception as e:
            self.step_text_var.set(f"Feature Perturbation Failed: {str(e)}")
            messagebox.showerror("Error", f"Feature Perturbation Failed: {str(e)}")
            return
        # 记录耗时
        self.feature_perturb_time = round(time.time() - start_time, 4)
        # 展示特征贡献度图
        # self.step_text_var.set("Step 2/5: Showing Feature Contribution Plot...")
        self.show_feature_contribution()
        self.progress["value"] = 40
        self.root.update_idletasks()

    def show_feature_contribution(self):
        """展示特征贡献度图表（独立保存 Figure 和 Canvas）"""
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 【新增】创建独立的 Figure
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))

        # 柱状图绘制（原逻辑不变）
        con_list = shared_globals.con_list
        feature_names = shared_globals.feature_names
        total_con = sum(con_list)
        if total_con == 0:
            self.step_text_var.set("No feature contributions to display")
            return
        con_list = [max(0.0, con) for con in con_list]
        total_con = sum(con_list)
        top_indices = np.argsort(con_list)[-10:][::-1]
        merged_con = [con_list[i] for i in top_indices]
        merged_names = [feature_names[i] for i in top_indices]
        other_con = total_con - sum(merged_con)
        other_con = max(0.0, other_con)
        merged_con.append(other_con)
        merged_names.append("others")

        axs[0].bar(merged_names, merged_con, alpha=0.6, color='#FF7F7F', edgecolor='#D62728')
        axs[0].set_title("Feature Contribution Values", fontsize=10, pad=10)

        axs[0].set_xticks(np.arange(len(merged_names)))  # 刻度位置与标签数量一致
        axs[0].set_xticklabels(merged_names, rotation=70, fontsize=8)  # 合并标签设置
        axs[0].set_ylabel("Contribution (JS Divergence)", fontsize=9)
        for i, bar in enumerate(axs[0].patches):
            height = bar.get_height()
            axs[0].text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=7
            )

        # 饼图绘制（原逻辑不变）
        axs[1].pie(merged_con, labels=merged_names, autopct='%1.1f%%', colors=plt.cm.Pastel1.colors,
                   textprops={'fontsize': 8})
        axs[1].set_title("Feature Contribution Distribution", fontsize=10, pad=10)
        plt.tight_layout()

        # 嵌入 Tkinter（首次显示时创建临时 Canvas）
        temp_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        temp_canvas.draw()
        temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 关键：只保存 (名称, Figure) 到历史记录，不保存 Canvas！
        self.plot_history.append(('Feature Contribution', fig))
        self.current_plot_index = len(self.plot_history) - 1
        self.update_nav_buttons()

        self.step_text_var.set("Step 3/5: Causal Graph Construction...")



    def start_causal_graph(self):
        """因果图构建（含耗时统计）"""
        start_time = time.time()
        try:
            # 初始化特征对象
            features = [Feature(
                name=n,
                feature_names=shared_globals.feature_names,
                all_con_values=shared_globals.con_list
            ) for n in shared_globals.all_names]
            shared_globals.features = features
            # 判断特征是否为数值型（选择相关系数类型）
            is_numeric = all(f.data.dtype.kind in 'iufc' for f in features)
            if is_numeric:
                shared_globals.corr_matrix = np.corrcoef([f.data for f in features])
            else:
                shared_globals.corr_matrix, _ = spearmanr([f.data for f in features])
            # 构建因果图
            valid_paths, dag, _ = build_causal_graph(
                features,
                shared_globals.corr_matrix,
                shared_globals.target_name,
                is_numeric
            )
            print(f"\nCausal Graph: Valid Paths Count = {len(valid_paths)}")
        except Exception as e:
            self.step_text_var.set(f"Causal Graph Failed: {str(e)}")
            messagebox.showerror("Error", f"Causal Graph Failed: {str(e)}")
            return
        # 记录耗时
        self.causal_graph_time = round(time.time() - start_time, 4)
        # 展示因果图
        # self.step_text_var.set("Step 3/5: Showing Causal Graph...")
        self.show_causal_graph(valid_paths, shared_globals.all_names)
        self.progress["value"] = 60  # 进度条：40%→60%
        self.root.update_idletasks()

    def show_causal_graph(self, valid_paths, all_names):
        """展示因果图（缩小尺寸）"""
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 创建有向图
        G = nx.DiGraph()
        G.add_nodes_from(all_names)  # 添加所有节点
        # 添加路径中的边
        for path in valid_paths:
            for i in range(len(path) - 1):
                u = all_names[path[i]]
                v = all_names[path[i + 1]]
                G.add_edge(u, v)

        # 获取特征对象（用于贡献值和噪声计算）
        features = [f for f in shared_globals.features if f.name in all_names]
        feature_dict = {f.name: f for f in features}

        # 处理可能的缺失特征
        missing_features = [name for name in all_names if name not in feature_dict]
        if missing_features:
            print(f"Warning: Missing Feature objects for {missing_features}, using default values")
            for name in missing_features:
                from causal_graph_build_v19_english import Feature  # 确保导入
                temp_feat = Feature(
                    name=name,
                    feature_names=shared_globals.feature_names,
                    all_con_values=shared_globals.con_list
                )
                temp_feat.con_i = 0.0
                temp_feat.noise = 0.0
                temp_feat.is_high_contrib = False
                feature_dict[name] = temp_feat

        # 节点布局（目标节点居中，其他节点环绕）
        target_node = shared_globals.target_name[0]
        pos = {target_node: (0, 0)}
        others = [n for n in all_names if n != target_node]
        angle = 2 * np.pi / len(others) if others else 0
        pos.update({
            others[i]: (2 * np.cos(i * angle), 2 * np.sin(i * angle))
            for i in range(len(others))
        })

        # 节点大小（基于贡献值）
        all_con = [f.con_i for f in features]
        max_con = max(all_con) if all_con and max(all_con) != 0 else 1.0
        node_size = [feature_dict[name].con_i / max_con * 2000 for name in all_names]

        # 节点颜色（高贡献特征为浅绿色，其他为浅蓝色）
        node_colors = [
            'lightgreen' if feature_dict[name].is_high_contrib else 'lightblue'
            for name in all_names
        ]

        # 边颜色（基于噪声值，使用RdYlGn colormap）
        edges = G.edges()
        edge_colors = []
        for u, v in edges:
            noise_u = feature_dict[u].noise
            noise_v = feature_dict[v].noise
            avg_noise = (noise_u + noise_v) / 2
            # 归一化噪声值到[0, 0.6]范围，映射到颜色（低噪声绿色，高噪声红色）
            normalized = min(avg_noise, 0.6) / 0.6
            edge_colors.append(plt.cm.RdYlGn(1 - normalized))

        # 创建图表（缩小尺寸）
        fig, ax = plt.subplots(figsize=(5, 5))  # 缩小尺寸

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='#000000',  # 黑色边框
            linewidths=1,
            ax=ax
        )

        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=edge_colors,
            arrowstyle='->',
            arrowsize=12,
            width=1.2,
            alpha=0.7,
            ax=ax
        )

        # 绘制节点标签（包含贡献值）
        nx.draw_networkx_labels(
            G, pos,
            labels={
                name: f"{name}\n{feature_dict[name].con_i:.5f}"
                for name in all_names
            },
            font_size=7,  # 缩小字体
            font_family='SimHei',  # 支持中文
            ax=ax
        )

        # 图表标题
        ax.set_title(
            "Causal Feature Graph (Green Nodes = High Contribution Features, Node Size = Contribution Value, Edge Color = Noise)",
            fontsize=9,
            pad=10
        )
        ax.axis('equal')  # 等比例显示
        ax.axis('off')  # 关闭坐标轴
        plt.tight_layout()

        # 嵌入 Tkinter（临时 Canvas）
        temp_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        temp_canvas.draw()
        temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 只保存 (名称, Figure)
        self.plot_history.append(('Causal Graph', fig))
        self.current_plot_index = len(self.plot_history) - 1
        self.update_nav_buttons()

        self.step_text_var.set("Step 4/5: High-potential Feature Recovery...")



    def start_feature_supply(self):
        """冗余特征恢复（含双特征扰动）"""
        start_time = time.time()
        try:
            # 初始化PathShapley模块并执行冗余检查
            shapley_module = PathShapleyModule(log_verbose=True, decay_type="exponential", alpha=0.7)
            shapley_module.run_redundant_check()
            # 打印冗余恢复结果（控制台调试用）
            print(f"\nHigh-potential Features:")
            print(f"  High-potential  Features: {shared_globals.redundant_features}")

        except Exception as e:
            self.step_text_var.set(f"High-potential Recovery Failed: {str(e)}")
            messagebox.showerror("Error", f"High-potential Recovery Failed: {str(e)}")
            return
        # 记录耗时
        self.feature_recover_time = round(time.time() - start_time, 4)
        # 展示冗余恢复结果
        # self.step_text_var.set("Step 4/5: Showing High-potential Recovery Result...")
        self.show_redundant_recover()
        self.progress["value"] = 80  # 进度条：60%→80%
        self.root.update_idletasks()

    def show_redundant_recover(self):
        """展示冗余恢复结果（左图居左、右图居右）"""
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 获取全局数据
        significant_logs = getattr(shared_globals, 'significant_feature_pairs_log', [])
        to_add_list = getattr(shared_globals, 'to_add', [])

        # --------------------------
        # 1. 子图布局：调整宽高比例与间距
        # --------------------------
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(
            1, 2,
            figure=fig,
            height_ratios=[1],
            width_ratios=[3, 1],  # 左侧更宽，容纳长文本
            wspace=0.4  # 增大水平间距，避免挤压
        )
        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1])

        # --------------------------
        # 2. 左侧：显著特征对日志（居左）
        # --------------------------
        if significant_logs:
            log_texts = [log["log_text"] for log in significant_logs]
            full_log_text = "\n\n".join(log_texts)
            ax_left.axis('off')
            ax_left.text(
                0.05, 0.95,  # 子图左侧起始位置
                f"Significant Feature Pairs Log (Total: {len(significant_logs)})\n{'-' * 80}\n{full_log_text}",
                ha='left', va='top',  # 文本左对齐、顶部对齐
                fontsize=7,
                fontfamily='SimHei',
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F8FF", edgecolor="#1890FF")
            )
        else:
            ax_left.axis('off')
            ax_left.text(
                0.5, 0.5,
                "No Significant Feature Pairs Log\n(No combinations meet the joint effect threshold)",
                ha='center', va='center',
                fontsize=9,
                fontfamily='SimHei',
                color="#696969",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", edgecolor="#CCCCCC")
            )
        # 左侧标题居左
        ax_left.set_title("Significant Feature Pairs Log", fontsize=10, pad=10, fontweight='bold', ha='left')

        # --------------------------
        # 3. 右侧：待恢复特征（居右）
        # --------------------------
        ax_right.axis('off')
        if to_add_list:
            to_add_text = "\n".join(to_add_list)
            ax_right.text(
                0.95, 0.95,  # 子图右侧起始位置
                f"To Recover Features\n{'-' * 40}\n{to_add_text}",
                ha='right', va='top',  # 文本右对齐、顶部对齐
                fontsize=7,
                fontfamily='SimHei',
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#E6E6FA", edgecolor="#6A5ACD")
            )
        else:
            ax_right.text(
                0.5, 0.5,
                "No Features in Recover List",
                ha='center', va='center',
                fontsize=9,
                fontfamily='SimHei',
                color="#696969",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", edgecolor="#CCCCCC")
            )
        # 右侧标题居右
        ax_right.set_title("Supply Features", fontsize=10, pad=10, fontweight='bold', ha='right')

        # --------------------------
        # 4. 自动优化整体布局
        # --------------------------
        plt.tight_layout()

        # --------------------------
        # 5. Tkinter 嵌入与导航更新
        # --------------------------
        temp_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        temp_canvas.draw()
        temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_history.append(('Significant Pairs Log', fig))
        self.current_plot_index = len(self.plot_history) - 1
        self.update_nav_buttons()

        self.step_text_var.set("Step 5/5: CauFeature Model Training...")


    def show_result(self):
        """展示最终结果（缩小尺寸）"""


        # 获取全局数据
        original_dim = self.original_feature_len
        final_dim = len(shared_globals.filtered_features)

        time_reduction = getattr(shared_globals, 'running_time', 0.0)
        acc_increase = getattr(shared_globals, 'accuracy', 0.0)
        total_time = self.feature_perturb_time + self.causal_graph_time + self.feature_recover_time

        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 构建结果文本
        result_text = (
            "=== CauFeature: Final Feature Selection Result ===\n\n"
            f"1. Feature Dimension Change\n"
            f"   Original Features: {original_dim}\n"
            f"   Filtered Features: {final_dim}\n"
            f"   Dimension Reduction Ratio: {((original_dim - final_dim) / original_dim * 100):.2f}%\n\n"
            f"2. Model Performance Optimization\n"
            f"   Time Reduction: {time_reduction:.2f} Seconds\n"
            f"   Accuracy Improvement: {acc_increase:.2f}%\n\n"
            f"3. Time Consumption Statistics\n"
            f"   Feature Perturbation: {self.feature_perturb_time:.2f}s\n"
            f"   Causal Graph Construction: {self.causal_graph_time:.2f}s\n"
            f"   Redundant Recovery: {self.feature_recover_time:.2f}s\n"
            f"   Total Time: {total_time:.2f}s\n\n"
            f"4. Final Filtered Features\n"
            f"   {', '.join(shared_globals.filtered_features) if shared_globals.filtered_features else 'No Valid Features'}"
        )

        # 创建文本展示图（缩小尺寸）
        fig, ax = plt.subplots(figsize=(8, 4))  # 缩小尺寸
        ax.axis('off')
        ax.text(
            0.05, 0.95,
            result_text,
            ha='left', va='top',
            fontsize=9,  # 缩小字体
            fontfamily='SimHei',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F8FF", edgecolor="#4169E1")
        )
        ax.set_title("Final Result Summary", fontsize=10, pad=10, fontweight='bold')

        # 嵌入 Tkinter（临时 Canvas）
        temp_canvas = FigureCanvasTkAgg(fig, master=self.frame)
        temp_canvas.draw()
        temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 只保存 (名称, Figure)
        self.plot_history.append(('Final Result', fig))
        self.current_plot_index = len(self.plot_history) - 1
        self.update_nav_buttons()

        # 更新步骤文本
        self.step_text_var.set("Feature Selection Completed Successfully!")
        self.progress["value"] = 100

        # 重新启用控件
        self.start_btn.config(state=tk.NORMAL)
        self.dataset_combobox.config(state="readonly")
        self.model_combobox.config(state="readonly")

    def update_nav_buttons(self):
        """更新导航按钮状态（带直观提示文本）"""
        total_plots = len(self.plot_history)  # 总图表数

        # 情况1：无图表或只有1个图表，禁用所有导航按钮
        if total_plots <= 1:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.prev_hint.config(text=f"(Current: {total_plots})")
            self.next_hint.config(text="(Click Untill CauFeature Completed)")
        # 情况2：有多个图表，根据当前索引判断按钮状态
        else:
            # Previous按钮：当前不是第1个图表时可点击
            prev_state = tk.NORMAL if self.current_plot_index > 0 else tk.DISABLED
            self.prev_btn.config(state=prev_state)

            # Next按钮：当前不是最后1个图表时可点击
            next_state = tk.NORMAL if self.current_plot_index < total_plots - 1 else tk.DISABLED
            self.next_btn.config(state=next_state)
            self.prev_hint.config(text=f"(Current: {self.current_plot_index + 1}/{total_plots})")
            self.next_hint.config(text=f"(Click Untill CauFeature Completed)")

    def show_previous_plot(self):
        """显示上一个图表"""
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self._display_current_plot()
            self.update_nav_buttons()

    def show_next_plot(self):
        """显示下一个图表"""
        if self.current_plot_index < len(self.plot_history) - 1:
            self.current_plot_index += 1
            self._display_current_plot()
            self.update_nav_buttons()

    def _display_current_plot(self):
        # 清空绘图区
        for widget in self.frame.winfo_children():
            widget.destroy()

        if self.current_plot_index < 0 or self.current_plot_index >= len(self.plot_history):
            return
        plot_name, current_fig = self.plot_history[self.current_plot_index]

        # 分类报告单独处理，调用show_model渲染界面
        if plot_name in ['Original Model Classification Report']:
            self.show_origin_model()
            self.step_text_var.set(f"Viewing: Original Model Classification Report")
        else:
            if plot_name in ['CauFeature Selected Model Classification Report']:
                self.show_after_model()
                self.step_text_var.set(f"Viewing: CauFeature Selected Model Classification Report")
            else:
                # 渲染matplotlib图表
                new_canvas = FigureCanvasTkAgg(current_fig, master=self.frame)
                new_canvas.draw()
                new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.step_text_var.set(f"Viewing: {plot_name}")







if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureSelectionApp(root)
    root.mainloop()