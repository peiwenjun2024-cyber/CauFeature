import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time  # 用于模拟进度条的时间流逝


class FeatureSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("特征选择工具")  # 窗口标题
        self.root.geometry("800x600")  # 窗口大小

        # ---------- 1. 步骤说明标签 ----------
        step_text = """1. 首先打开工具，选择数据集和模型（今天用 CNN）；
2. 点「开始特征选择」—— 能看到它先跑自适应扰动（进度条：0%→100%）；
3. 接着它会画出特征贡献度 —— 看到那些红色柱子了吗？这就是要保留的高影响特征；
4. 然后因果图弹出来 —— 看这些箭头！只有因果关系能留下来（没有虚假关联）；
5. 最后它会加进协同特征，输出最终特征集。搞定！5 个特征，82.36% 准确率，就这么简单！"""
        self.step_label = tk.Label(
            root,
            text=step_text,
            justify=tk.LEFT,  # 文字左对齐
            wraplength=700  # 文字超过700像素自动换行
        )
        self.step_label.pack(pady=10)  # 上下留10像素间距

        # ---------- 2. “开始特征选择”按钮 ----------
        self.start_btn = tk.Button(
            root,
            text="开始特征选择",
            command=self.start_feature_selection  # 点击触发的函数
        )
        self.start_btn.pack(pady=10)

        # ---------- 3. 进度条（模拟“自适应扰动”过程） ----------
        self.progress = ttk.Progressbar(
            root,
            orient=tk.HORIZONTAL,  # 水平方向
            length=400,  # 进度条长度
            mode='determinate'  # 确定式（从0到100）
        )
        self.progress.pack(pady=10)

        # ---------- 4. 绘图框架（用于显示“特征贡献度”和“因果图”） ----------
        self.frame = tk.Frame(root)
        self.frame.pack(pady=10, fill=tk.BOTH, expand=True)  # 填充并拉伸

        # ---------- 5. 结果标签（显示最终特征数和准确率） ----------
        self.result_label = tk.Label(
            root,
            text="",
            font=("Arial", 12)  # 字体和大小
        )
        self.result_label.pack(pady=10)

    def start_feature_selection(self):
        """点击“开始特征选择”后，执行的逻辑：进度条→特征贡献度→因果图→结果"""
        # 禁用按钮（防止重复点击）
        self.start_btn.config(state=tk.DISABLED)

        # 模拟进度条（从0%到100%）
        for i in range(101):
            self.progress["value"] = i
            self.root.update_idletasks()  # 强制更新UI
            time.sleep(0.02)  # 暂停0.02秒，模拟进度条动画

        # 进度条完成后，显示「特征贡献度」图
        self.show_feature_contribution()

    def show_feature_contribution(self):
        """显示“特征贡献度”柱状图"""
        # 清空绘图框架内的旧组件
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 创建matplotlib图表
        fig, ax = plt.subplots(figsize=(6, 4))
        features = ["特征1", "特征2", "特征3", "特征4", "特征5", "特征6", "特征7"]
        contributions = np.random.rand(len(features)) * 100  # 模拟特征贡献度（0-100随机值）
        ax.bar(features, contributions, color='red')  # 红色柱子
        ax.set_title("特征贡献度")
        ax.set_ylabel("贡献度 (%)")

        # 将matplotlib图表嵌入tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # 2秒后，显示「因果图」（通过after方法延迟执行）
        self.root.after(2000, self.show_causal_graph)

    def show_causal_graph(self):
        """显示“因果图”（用箭头模拟因果关系）"""
        # 清空绘图框架内的旧组件
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 创建matplotlib图表（画箭头模拟因果关系）
        fig, ax = plt.subplots(figsize=(6, 4))
        # 画因果箭头
        ax.plot([0.2, 0.4], [0.5, 0.5], '->', linewidth=2, color='blue')
        ax.plot([0.4, 0.6], [0.5, 0.7], '->', linewidth=2, color='blue')
        ax.plot([0.4, 0.6], [0.5, 0.3], '->', linewidth=2, color='blue')
        # 标注节点
        ax.text(0.15, 0.5, "特征A", fontsize=12)
        ax.text(0.35, 0.5, "特征B", fontsize=12)
        ax.text(0.55, 0.7, "特征C", fontsize=12)
        ax.text(0.55, 0.3, "特征D", fontsize=12)
        ax.set_title("因果图")
        ax.set_xlim(0, 1)  # x轴范围
        ax.set_ylim(0, 1)  # y轴范围
        ax.axis('off')  # 隐藏坐标轴

        # 将matplotlib图表嵌入tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # 2秒后，显示「最终结果」
        self.root.after(2000, self.show_result)

    def show_result(self):
        """显示最终结果：特征数 + 准确率"""
        # 清空绘图框架内的旧组件
        for widget in self.frame.winfo_children():
            widget.destroy()

        # 显示结果标签
        self.result_label.config(text="最终特征集：5 个特征，准确率 82.36%")
        # 重新启用“开始特征选择”按钮
        self.start_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()  # 创建根窗口
    app = FeatureSelectionApp(root)  # 初始化应用
    root.mainloop()  # 启动主事件循环