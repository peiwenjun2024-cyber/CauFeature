import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, gaussian_kde
from collections import defaultdict
import shared_globals
from multiprocessing import Lock
from scipy.ndimage import gaussian_filter1d


def get_and_print_high_contrib_features(features, print_lock):
    """提取高贡献特征名称并打印"""
    high_contrib_names = [f.name for f in features if f.is_high_contrib]

    with print_lock:
        print(f"\n【高贡献特征列表】共{len(high_contrib_names)}个:")
        for i, name in enumerate(high_contrib_names, 1):
            print(f"{i}. {name} (贡献值: {[f.con_i for f in features if f.name == name][0]:.4f})")

    return high_contrib_names


class CausalConfig:
    def __init__(self):
        # 保留高贡献特征判定核心参数
        self.high_contrib_method = "data_driven"
        self.small_sample_threshold = 10
        self.high_contrib_fallback_percentile = 75
        self.con_sigma = 0.5  # 用于噪声相关计算


config = CausalConfig()
print_lock = Lock()


class Feature:
    """保留核心属性：贡献值、噪声、高贡献判定"""

    def __init__(self, name: str, feature_names: list, all_con_values=None):
        self.name = name
        self.data = shared_globals.data[name].values  # 原始数据

        # 计算噪声（变异系数，保留核心逻辑）
        try:
            self.data = np.asarray(self.data, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"特征 {name} 包含非数值数据: {e}") from e

        mean = np.mean(self.data)
        std = np.std(self.data)

        # 处理常量特征（避免除数为零）
        if std < 1e-9:
            self.noise = 0.0
            with print_lock:
                print(f"特征 {name} 为常量特征，噪声强制设为0")
        else:
            denom = np.abs(mean) + 1e-6  # 增强分母保护，解决除数为零问题
            self.noise = std / denom
            if self.noise > 1e6:  # 截断极端值
                self.noise = 1e6
                with print_lock:
                    print(f"特征 {name} 噪声值过大，已截断至1e6")

        # 贡献值
        self.con_i = (shared_globals.con_list[feature_names.index(name)]
                      if name in feature_names else 0.0)
        self.all_con_values = all_con_values

        # 保留高贡献特征判定逻辑
        self.is_high_contrib = False
        self.threshold = None
        self.judgment_reason = ""

        if feature_names and name in feature_names and self.all_con_values is not None:
            # 数据驱动的高贡献判定（核心保留）
            if len(self.all_con_values) >= config.small_sample_threshold:
                self.is_high_contrib, self.threshold, self.judgment_reason = self._is_high_contrib_data_driven(
                    self.all_con_values)
            else:
                # 小样本分位数判定
                threshold = np.percentile(self.all_con_values, 80)
                self.threshold = threshold
                self.is_high_contrib = self.con_i >= threshold
                self.judgment_reason = f"小样本（{len(self.all_con_values)} < {config.small_sample_threshold}），使用80%分位数阈值: {threshold:.4f}"

    def _is_high_contrib_data_driven(self, all_con_values):
        """保留基于密度拐点的高贡献判定"""
        kde = gaussian_kde(all_con_values, bw_method='silverman')
        sorted_con = np.sort(all_con_values)
        density = kde(sorted_con)
        density_smoothed = gaussian_filter1d(density, sigma=1)  # 平滑处理

        # 计算二阶导数找拐点
        first_deriv = np.gradient(density_smoothed)
        second_deriv = np.gradient(first_deriv)
        sign_changes = np.diff(np.sign(second_deriv))
        inflection_points = np.where(sign_changes != 0)[0]

        # 右侧拐点判定
        if len(inflection_points) > 0:
            right_inflections = [i for i in inflection_points if i > len(sorted_con) * 0.3]
            if right_inflections:
                threshold_idx = right_inflections[-1]
                threshold = sorted_con[threshold_idx]
                max_con = max(all_con_values)
                adjusted_threshold = min(threshold, max_con * 0.9)
                reason = f"使用右侧拐点阈值: {threshold:.4f}，调整后: {adjusted_threshold:.4f}"
                return self.con_i >= adjusted_threshold, adjusted_threshold, reason

        # 兜底分位数判定
        threshold = np.percentile(all_con_values, config.high_contrib_fallback_percentile)
        reason = f"未找到有效拐点，使用{config.high_contrib_fallback_percentile}%分位数阈值: {threshold:.4f}"
        return self.con_i >= threshold, threshold, reason

    def copy_with_data(self, new_data):
        """保留数据复制方法"""
        new_feature = Feature(self.name, [], None)
        new_feature.data = new_data
        new_feature.noise = self.noise
        new_feature.con_i = self.con_i
        new_feature.is_high_contrib = self.is_high_contrib
        new_feature.threshold = self.threshold
        new_feature.judgment_reason = self.judgment_reason
        return new_feature


def draw_causal_graph(valid_paths, features, all_names, target_name):
    """保留所有特征展示，突出高贡献特征和兜底路径，保留噪声信息"""
    G = nx.DiGraph()
    target_node = target_name[0]
    G.add_nodes_from(all_names)  # 展示所有特征

    # 只添加兜底路径
    for path in valid_paths:
        for i in range(len(path) - 1):
            G.add_edge(all_names[path[i]], all_names[path[i + 1]])

    # 布局：目标在中心，其他特征环绕
    pos = {target_node: (0, 0)}
    others = [n for n in all_names if n != target_node]
    angle = 2 * np.pi / len(others) if others else 0
    pos.update({others[i]: (2 * np.cos(i * angle), 2 * np.sin(i * angle)) for i in range(len(others))})

    # 节点大小基于贡献值，颜色区分高贡献特征
    all_con = [f.con_i for f in features]
    max_con = max(all_con) if all_con and max(all_con) != 0 else 1.0
    node_size = [f.con_i / max_con * 2000 for f in features]
    node_colors = ['lightgreen' if f.is_high_contrib else 'lightblue' for f in features]

    # 边颜色基于噪声（保留噪声特征）
    edges = G.edges()
    edge_colors = [(features[all_names.index(u)].noise + features[all_names.index(v)].noise) / 2
                   for u, v in edges]
    edge_colors = [plt.cm.RdYlGn(1 - min(n, 0.6) / 0.6) for n in edge_colors]  # 噪声越低越绿

    # 绘制图形
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                           arrowstyle='->', width=2, alpha=0.8)
    nx.draw_networkx_labels(G, pos,
                            labels={n: f"{n}\n{features[i].con_i:.5f}"
                                    for i, n in enumerate(all_names)},
                            font_size=9)
    plt.title("因果特征图（绿色=高贡献特征，节点大小=贡献值，边颜色=噪声）")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def classify_paths(valid_paths, all_names, target_name):
    """简化路径分类，只保留直接路径"""
    direct = []
    for path in valid_paths:
        path_str = " → ".join([all_names[i] for i in path])
        direct.append(path_str)

    shared_globals.direct_paths = direct
    shared_globals.indirect_paths = []

    with print_lock:
        print("\n【直接路径（最高贡献特征→目标）】")
        for i, p in enumerate(direct, 1):
            print(f"{i}. {p}")


def build_causal_graph(features, corr_matrix, target_name, is_numeric):
    """简化核心流程：只保留高贡献判定和兜底路径"""
    # 获取目标索引
    target_idx = shared_globals.all_names.index(target_name[0])
    shared_globals.target_idx = target_idx

    # 筛选非目标特征
    non_target_features = [f for f in features if f.name != target_name[0]]
    if not non_target_features:
        raise ValueError("没有找到非目标变量的特征")

    # 保留兜底逻辑：最大贡献特征指向目标
    max_con_idx = max(range(len(features)), key=lambda x: features[x].con_i if x != target_idx else -1)
    max_con_feature = features[max_con_idx]
    fallback_path = [max_con_idx, target_idx]
    best_paths = [fallback_path]

    with print_lock:
        print(f"\n兜底路径: {shared_globals.all_names[max_con_idx]} -> {target_name[0]} "
              f"(贡献值: {max_con_feature.con_i:.4f}, 噪声: {max_con_feature.noise:.4f})")

    # 提取高贡献特征阈值
    high_contrib_thresholds = [f.threshold for f in features if f.is_high_contrib]
    overall_high_contrib_threshold = np.mean(high_contrib_thresholds) if high_contrib_thresholds else 0

    # 记录涉及特征
    # 关键修改：将高贡献特征写入全局变量filtered_features
    # 仅保留最高贡献的那一个特征到filtered_features
    filtered_features = [max_con_feature.name]  # 只包含最高贡献特征
    shared_globals.filtered_features = filtered_features  # 写入全局变量

    with print_lock:
        print(f"\n【所有特征列表】共{len(filtered_features)}个:")
        for i, feat in enumerate(filtered_features, 1):
            print(f"{i}. {feat}")

    # 绘图和路径分类
    draw_causal_graph(best_paths, features, shared_globals.all_names, target_name)
    classify_paths(best_paths, shared_globals.all_names, target_name)
    shared_globals.valid_paths = best_paths

    # 结果摘要
    with print_lock:
        print("\n" + "=" * 60)
        print("           因果图构建最终结果")
        print("=" * 60)
        print(f"1. 兜底路径: {shared_globals.all_names[max_con_idx]} -> {target_name[0]}")
        print(f"2. 最高贡献特征贡献值: {max_con_feature.con_i:.4f}")
        print(f"3. 最高贡献特征噪声: {max_con_feature.noise:.4f}")
        print(f"4. 高贡献特征判定阈值 (平均): {overall_high_contrib_threshold:.4f}")
        print(f"5. 高贡献特征数量: {sum(1 for f in features if f.is_high_contrib)}")
        print("=" * 60 + "\n")

    # 输出高贡献特征
    high_contrib_features = get_and_print_high_contrib_features(features, print_lock)
    shared_globals.high_contrib_features = high_contrib_features

    return best_paths, None, shared_globals.all_names


if __name__ == '__main__':
    if not hasattr(shared_globals, 'con_list'):
        raise ValueError("请先运行特征扰动模块，确保shared_globals包含con_list")

    # 预计算所有贡献值
    all_con_values = [shared_globals.con_list[i] for i in range(len(shared_globals.feature_names))]
    with print_lock:
        print(
            f"\n所有特征贡献值列表: {[(shared_globals.feature_names[i], round(all_con_values[i], 4)) for i in range(len(all_con_values))]}")
        print(f"贡献值样本量: {len(all_con_values)}, 小样本阈值: {config.small_sample_threshold}")

    # 初始化特征（保留高贡献判定）
    features = [Feature(name=n, feature_names=shared_globals.feature_names, all_con_values=all_con_values)
                for n in shared_globals.all_names]

    # 数据类型检查
    is_numeric = all(f.data.dtype.kind in 'iufc' for f in features)
    with print_lock:
        print(f"\n数据类型检查: {'数值型' if is_numeric else '非数值型'}")

    # 计算相关矩阵（仅为兼容性保留，不影响核心逻辑）
    if is_numeric:
        corr_matrix = np.corrcoef([f.data for f in features])
    else:
        corr_matrix, _ = spearmanr([f.data for f in features])

    # 构建因果图
    valid_paths, _, _ = build_causal_graph(features, corr_matrix,
                                           shared_globals.target_name,
                                           is_numeric)