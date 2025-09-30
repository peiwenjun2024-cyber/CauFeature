#适配多分类场景，v5版本试图采用js散度作为贡献值，以下优势
# '''避免极端类别干扰
# 多分类中可能存在 “稀有类别”（如某类别样本仅占 1%），KL 散度可能因这类别的概率微小变化（如从 0.01 变为 0.02）而产生较大数值，高估特征对整体分类的影响。JS 散度因取平均分布，对稀有类别的极端变化更稳健。适配类别数动态变化的场景
# 若后续扩展模型至更多类别（如从 6 类扩展到 10 类），JS 散度的有界性使其贡献值范围保持稳定（仍在\([0, \log 2]\)），便于前后实验结果对比；而 KL 散度的取值范围会随类别数增加而扩大，破坏实验一致性。与多分类评估指标兼容
# 多分类常用的评估指标（如宏平均 F1、微平均 F1）均为有界值（\([0,1]\)），JS 散度的有界性使其可与这些指标直接关联（如 “贡献值 0.5 的特征对宏 F1 的提升显著”），便于业务解释。'''

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
import os

import shared_globals

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import time
from typing import List, Tuple
from tensorflow.keras.models import load_model
from scipy.stats import entropy  # 提前导入entropy，避免重复导入


class CausalFeaturePerturbation:
    """适配CausalFeatureX框架的特征扰动模块（并行计算优化版）"""

    def __init__(self, model: tf.keras.Model, data: pd.DataFrame):
        self.model = model  # 待评估模型
        self.data = data  # 输入数据集
        self.sample_size = data.shape[0]  # 样本量
        self.baseline = self._calculate_baseline()  # 基线输入
        self.baseline_np = self.baseline.values.astype(np.float32).copy()
        self.n_features = len(shared_globals.feature_names)

        # 重塑基线数据为三维张量
        self.baseline_reshaped = self.baseline_np.reshape(-1, self.n_features, 1)
        # 多分类：基线预测为概率分布向量（保留完整分布）
        self.baseline_pred = self.model.predict(self.baseline_reshaped, verbose=0)
        # 修正：基线分布应为二维数组中的第一行（形状为(num_classes,)）
        self.baseline_pred_dist = self.baseline_pred[0]  # 关键修改：取第一个样本的完整分布

        # 预计算特征值
        self.feature_values = {}
        for col in shared_globals.feature_names:
            try:
                self.feature_values[col] = sorted(pd.to_numeric(data[col], errors='coerce').dropna().unique().tolist())
            except:
                self.feature_values[col] = []

        shared_globals.baselines = self.baseline
        shared_globals.baseline_preds = self.baseline_pred

    def _calculate_baseline(self):
        """计算基线值"""
        baseline = pd.Series(index=shared_globals.feature_names, dtype=np.float32)
        for feature in shared_globals.feature_names:
            baseline[feature] = shared_globals.feature_data[feature].mean()
        return pd.DataFrame([baseline])

    def get_col_ranges(self, col_name: str) -> Tuple[float, float, int, List[float]]:
        """优化后的特征统计计算"""
        s = shared_globals.data[col_name]
        numeric_series = pd.to_numeric(s, errors='coerce').dropna()

        if numeric_series.empty:
            return (0.0, 0.0, 0, [])

        min_val = numeric_series.min()
        max_val = numeric_series.max()
        unique_vals = numeric_series.unique()

        return (
            min_val,
            max_val,
            len(unique_vals),
            sorted(unique_vals.tolist())
        )

    def value_range(self):
        """并行计算所有特征的统计范围"""
        features = list(shared_globals.feature_data.columns)
        results = [None] * len(features)

        def process_feature(col):
            return self.get_col_ranges(col)

        # 合理设置线程数，通常为CPU核心数的2-4倍
        max_workers = os.cpu_count() * 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_feature, col): i for i, col in enumerate(features)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"特征 {features[idx]} 处理失败: {e}")
                    results[idx] = (0.0, 0.0, 0, [])

        min_list = [res[0] for res in results]
        max_list = [res[1] for res in results]
        count_list = [res[2] for res in results]
        range_list = [res[3] for res in results]
        return min_list, max_list, count_list, range_list

    def disturbance_range(self, min_list: List[float], max_list: List[float],
                          count_list: List[int], range_list: List[List[float]]) -> List[List[float]]:
        """核心扰动值生成函数"""
        x_list = []
        HIGH_DIVERSITY_THRESHOLD = 0.9
        LARGE_DIFF_THRESHOLD = 10000
        NUM_PARTITIONS = 100

        for j, col in tqdm(enumerate(shared_globals.feature_names),
                           desc="生成特征扰动值",
                           total=len(shared_globals.feature_names)):
            min_val, max_val = min_list[j], max_list[j]
            unique_count, val_range = count_list[j], range_list[j]

            if min_val > max_val:
                x_list.append([])
                continue

            diff = max_val - min_val
            diversity_ratio = unique_count / self.sample_size if self.sample_size > 0 else 0

            if diff >= LARGE_DIFF_THRESHOLD:
                if diversity_ratio >= HIGH_DIVERSITY_THRESHOLD:
                    perturb_values = np.linspace(min_val, max_val, NUM_PARTITIONS + 1, dtype=np.float32)
                    x_list.append(np.round(perturb_values, 6).tolist())
                else:
                    x_list.append(sorted(val_range))
            else:
                if diff <= 1 and any(not float(v).is_integer() for v in val_range):
                    perturb_values = np.linspace(min_val, max_val, unique_count, dtype=np.float32)
                    x_list.append(np.round(perturb_values, 6).tolist())
                else:
                    x_list.append(list(range(int(min_val), int(max_val) + 1)))

        return x_list

    def _js_divergence(self, p, q):
        """计算两个概率分布之间的JS散度"""
        # 数值稳定性处理：限制概率值范围，避免log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        # 计算平均分布
        m = 0.5 * (p + q)
        # 计算JS散度（0.5*(KL(p||m) + KL(q||m))）
        return 0.5 * (entropy(p, m) + entropy(q, m))

    def _process_single_feature(self, j, col, values):
        """处理单个特征的贡献值计算，使用JS散度"""
        try:
            # 筛选有效数值扰动值
            valid_values = []
            for val in values:
                try:
                    valid_values.append(float(val))
                except (ValueError, TypeError):
                    continue

            if not valid_values:
                return j, [], 0.0

            # 批量构建输入
            batch_input = np.repeat(self.baseline_np, len(valid_values), axis=0)
            batch_input[:, j] = valid_values
            batch_input = batch_input.reshape(-1, self.n_features, 1).astype(np.float32)
            batch_tensor = tf.convert_to_tensor(batch_input)

            # 模型预测（保留完整分布，形状为(n_samples, num_classes)）
            preds = self.model.predict(batch_tensor, verbose=0)

            # 计算与基线分布的JS散度
            valid_diffs = []
            for dist in preds:  # dist是每个扰动值对应的完整概率分布
                js_div = self._js_divergence(dist, self.baseline_pred_dist)
                valid_diffs.append(js_div)

            # 贡献值定义为最大JS散度
            con = max(valid_diffs) if valid_diffs else 0.0
            return j, valid_diffs, con

        except Exception as e:
            print(f"特征 {col} 处理出错: {str(e)}")
            return j, [], 0.0

    def calculate_contributions(self, x_list: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        """并行计算特征贡献值（使用JS散度）"""
        y_list = [[] for _ in shared_globals.feature_names]
        con_list = [0.0 for _ in shared_globals.feature_names]
        start = time.perf_counter()

        features = shared_globals.feature_names
        max_workers = min(8, os.cpu_count())
        print(f"使用 {max_workers} 个线程并行计算特征贡献值...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for j, col in enumerate(features):
                values = x_list[j]
                if values:
                    futures.append(executor.submit(
                        self._process_single_feature,
                        j, col, values
                    ))

            for future in tqdm(as_completed(futures), total=len(futures), desc="计算特征贡献值"):
                j, y, con = future.result()
                y_list[j] = y
                con_list[j] = con

        end = time.perf_counter()
        print(f"特征扰动实验总耗时：{end - start:.2f}秒")
        return y_list, con_list

    def calculate_joint_contributions(self, target_features, max_combinations=10000):
        """计算多特征联合干预的最大JS散度"""
        if not target_features:
            return 0.0

        feat_indices = [shared_globals.feature_names.index(f) for f in target_features]
        feat_perturb = [self.perturb_values[f] for f in target_features]

        total_combinations = np.prod([len(p) for p in feat_perturb])
        if total_combinations > max_combinations:
            print(f"组合数过多（{total_combinations}），采样{max_combinations}个组合")
            sampled_combinations = []
            for _ in range(max_combinations):

                combo = [np.random.choice(p) for p in feat_perturb]
                sampled_combinations.append(combo)
        else:
            from itertools import product
            sampled_combinations = list(product(*feat_perturb))

        # 批量构建输入并预测
        batch_input = np.repeat(self.baseline_np, len(sampled_combinations), axis=0)
        for i, idx in enumerate(feat_indices):
            values = [combo[i] for combo in sampled_combinations]
            batch_input[:, idx] = values

        # 关键修改：根据当前batch_input的列数动态计算特征数
        current_n_features = batch_input.shape[1]  # 代替self.n_features
        batch_input_reshaped = batch_input.reshape(-1, current_n_features, 1).astype(np.float32)  # 动态重塑

        preds = self.model.predict(batch_input_reshaped, verbose=0)  # 形状为(n_samples, num_classes)

        # 计算最大JS散度
        max_js = 0.0
        for dist in preds:
            js_div = self._js_divergence(dist, self.baseline_pred_dist)
            if js_div > max_js:
                max_js = js_div
        return max_js

    def visualization(self, x_list: List[List[float]], y_list: List[List[float]],
                      con_list: List[float]) -> List[float]:
        """可视化特征扰动效应（JS散度是非负的，无需取绝对值）"""
        max_abs_values = [0.0] * len(shared_globals.feature_names)

        top_indices = np.argsort(con_list)[-20:][::-1]
        for i in top_indices:
            col = shared_globals.feature_names[i]
            if not y_list[i]:
                continue

            max_abs = max(y_list[i])  # JS散度非负，直接取最大值
            max_abs_values[i] = max_abs

        for i in range(len(shared_globals.feature_names)):
            if i not in top_indices and y_list[i]:
                max_abs_values[i] = max(y_list[i])

        return max_abs_values

    def feature_important(self, con_list: List[float]):
        """特征重要性分析（使用JS散度结果）"""
        total_con = sum(con_list)
        if total_con == 0:
            return

        # 确保贡献值非负（JS散度理论上非负，但可能因计算误差出现微小负值）
        con_list = [max(0.0, con) for con in con_list]  # 确保非负
        total_con = sum(con_list)  # 重新计算总和，避免误差累积

        top_indices = np.argsort(con_list)[-10:][::-1]
        merged_con = [con_list[i] for i in top_indices]
        merged_names = [shared_globals.feature_names[i] for i in top_indices]
        other_con = total_con - sum(merged_con)
        other_con = max(0.0, other_con)  # 关键修复：避免负数

        merged_con.append(other_con)
        merged_names.append("others")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(merged_names, merged_con, alpha=0.6)
        axs[0].tick_params(axis='x', rotation=70)
        axs[1].pie(merged_con, labels=merged_names, autopct='%1.1f%%')
        plt.tight_layout()
        plt.show()

    def run_perturbation(self):
        """整合所有步骤"""
        min_list, max_list, count_list, range_list = self.value_range()
        x_list = self.disturbance_range(min_list, max_list, count_list, range_list)
        self.perturb_values = {
            shared_globals.feature_names[i]: x_list[i]
            for i in range(len(shared_globals.feature_names))
        }

        # 计算贡献值（使用JS散度）
        y_list, con_list = self.calculate_contributions(x_list)

        max_abs_values = self.visualization(x_list, y_list, con_list)
        self.feature_important(con_list)

        # 保存结果
        self.feature_max_abs = [(shared_globals.feature_names[i], max_abs_values[i])
                                for i in range(len(shared_globals.feature_names))]
        self.feature_max_abs.sort(key=lambda x: x[1], reverse=True)
        self.con_list = con_list

        # 更新全局变量
        shared_globals.con_list = self.con_list
        shared_globals.feature_max_abs = self.feature_max_abs
        shared_globals.x_list = x_list
        shared_globals.feature_values = self.feature_values
        shared_globals.perturbation_instance = self


    def initialize_original_preds(self,data: pd.DataFrame, model) -> None:
        """
        计算原始样本的预测值、基线分布及JS散度并存入全局变量

        参数:
            data: 原始数据集（包含特征列和目标列）
            model: 训练好的模型（用于预测）
        """
        # 1. 提取特征列（排除目标列）
        feature_data = shared_globals.feature_data  # DataFrame格式，仅包含特征
        print(f"特征数据形状: {feature_data.shape}")

        # 2. 适配模型输入形状（1D CNN需要三维输入：(样本数, 特征数, 1)）
        features_np = feature_data.values  # 转换为numpy数组 (样本数, 特征数)
        features_reshaped = features_np.reshape(-1, features_np.shape[1], 1)  # 添加通道维度

        # 3. 计算原始样本的预测值（概率分布）
        try:
            original_preds = model.predict(features_reshaped, verbose=0)
            # 确保预测值为二维数组（样本数, 类别数）
            if original_preds.ndim == 1:
                original_preds = original_preds.reshape(-1, 1)
        except Exception as e:
            raise RuntimeError(f"计算原始样本预测值失败：{e}") from e

        # 4. 计算基线分布（所有样本预测的平均分布）
        # 对于多分类：每个类别取平均概率；对于二分类：保持分布形式
        baseline_dist = np.mean(original_preds, axis=0)
        # 确保基线分布是有效的概率分布（和为1）
        baseline_dist = baseline_dist / np.sum(baseline_dist)

        # 5. 计算每个样本预测分布与基线分布的JS散度
        sample_js_divergences = []
        for pred_dist in original_preds:
            js = self._js_divergence(pred_dist, baseline_dist)
            sample_js_divergences.append(js)
        sample_js_divergences = np.array(sample_js_divergences)

        # 6. 存入全局变量
        shared_globals.original_preds = original_preds  # 每个样本的预测分布
        shared_globals.baseline_dist = baseline_dist  # 基线分布（全体样本平均）
        shared_globals.sample_js = sample_js_divergences  # 每个样本与基线的JS散度
        shared_globals.mean_js = np.mean(sample_js_divergences)  # 平均JS散度

        print(f"已将原始样本预测值存入全局变量，形状: {original_preds.shape}")
        print(f"基线分布形状: {baseline_dist.shape}, 平均JS散度: {np.mean(sample_js_divergences):.4f}")

if __name__ == "__main__":
    shared_globals.init()
    model = load_model("model/cnn_model.keras")
    df = pd.read_csv("data/heart.csv")

    shared_globals.feature_names = df.columns[:-1].tolist()
    shared_globals.feature_data = df[shared_globals.feature_names]

    perturb = CausalFeaturePerturbation(model, df)
    perturb.run_perturbation()

