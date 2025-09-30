#试图解决除数为零问题
import os
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import norm, spearmanr, gaussian_kde
from itertools import combinations
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
import shared_globals
from multiprocessing import Pool, Lock  # 改为multiprocessing.Lock
import threading
#替换基于线性相关的独立性检验，采用非线性因果发现算法，如 ANM（Additive Noise Models，加性噪声模型）或 KCIT（Kernel Conditional Independence Test，核条件独立性检验）。

def get_and_print_high_contrib_features(features, print_lock):
    """提取高贡献特征名称并打印（仅输出名称）"""
    # 提取名称列表
    high_contrib_names = [f.name for f in features if f.is_high_contrib]

    # 打印（带锁确保多进程安全）
    with print_lock:
        print(f"\n【High Contribution Feature List】total sum:{len(high_contrib_names)}:")
        for i, name in enumerate(high_contrib_names, 1):
            print(f"{i}. {name} (Contribution Value: {[f.con_i for f in features if f.name == name][0]:.4f})")  # 输出名称和贡献值

    return high_contrib_names


# 全局配置类 - 将阈值集中管理并提供解释
class CausalConfig:
    def __init__(self):
        # 可配置参数（附带说明）
        self.high_contrib_method = "data_driven"  # 数据驱动的高贡献特征判定
        self.alpha_adjust_strategy = "adaptive"  # 自适应alpha调整
        self.stability_threshold_method = "fdr"  # 基于错误发现率的稳定性阈值
        self.auto_tune_parameters = True  # 新增：自动参数调整开关

        # 保留少量必要的超参数（而非硬编码）
        self.max_alpha_reduction = 0.5  # alpha最大缩减比例
        self.min_stability_threshold = 0.2  # 最小稳定性阈值
        self.max_cond_reduction = 1  # 条件组合最大减少量
        self.fdr_control_level = 0.2  # FDR控制水平（假阳性率上限）
        self.small_sample_threshold = 10  # 小样本判定阈值
        self.con_sigma = 0.5  # 贡献值差异惩罚的标准差
        self.high_contrib_fallback_percentile = 75  # 新增：fallback分位数，可调整
        self.n_bootstrap = 30  # Bootstrap抽样次数
        self.con_percentile = 30  # 贡献值分位数阈值
        self.noise_percentile = 70  # 噪声分位数阈值
        self.min_valid_paths = 1  # 最小有效路径数

    def auto_tune_based_on_data(self, features):
        """基于数据特征自动调整参数"""
        if not self.auto_tune_parameters:
            return

        # 基于特征数量自动调整FDR控制水平
        n_features = len(features)
        self.fdr_control_level = max(0.05, min(0.3, 0.2 * (1 + np.log10(n_features / 10))))

        # 基于特征维度自动调整最大路径长度
        if n_features < 10:
            self.max_path_length = 2
        elif n_features < 50:
            self.max_path_length = 3
        else:
            self.max_path_length = 4  # 特征多的时候允许更长路径

        # 基于数据噪声水平调整最小稳定性阈值
        avg_noise = np.mean([f.noise for f in features])
        self.min_stability_threshold = max(0.1, min(0.3, 0.2 * (1 - avg_noise)))

    def get_parameter_explanations(self):
        """返回所有关键参数的详细说明 - 修正版本"""
        # 直接构建完整的解释字典，避免依赖super()
        return {
            # 方法选择参数
            "high_contrib_method": {
                "值": self.high_contrib_method,
                "说明": "高贡献特征判定方法：'data_driven'表示基于贡献值分布的密度拐点自动确定阈值，避免人为比例设定"
            },
            "alpha_adjust_strategy": {
                "值": self.alpha_adjust_strategy,
                "说明": "alpha调整策略：'adaptive'表示根据特征贡献值动态调整，贡献值越高调整幅度越大"
            },
            "stability_threshold_method": {
                "值": self.stability_threshold_method,
                "说明": "路径稳定性阈值计算方法：'fdr'表示使用Benjamini-Hochberg方法控制错误发现率"
            },

            # 数值参数
            "max_alpha_reduction": {
                "值": self.max_alpha_reduction,
                "说明": "alpha最大缩减比例，控制高贡献特征对的独立性检验阈值调整上限"
            },
            "min_stability_threshold": {
                "值": self.min_stability_threshold,
                "说明": "路径稳定性的最小阈值，即使FDR计算不显著，也保留超过此阈值的路径"
            },
            "max_cond_reduction": {
                "值": self.max_cond_reduction,
                "说明": "条件组合的最大减少量，高贡献特征对在骨架构建时最多减少的条件组合数量"
            },
            "fdr_control_level": {
                "值": self.fdr_control_level,
                "说明": "错误发现率控制水平，控制路径选择中的假阳性比例上限"
            },
            "small_sample_threshold": {
                "值": self.small_sample_threshold,
                "说明": "小样本判定阈值，当贡献值样本量小于此值时，切换到更保守的高贡献判定方法"
            },
            "con_sigma": {
                "值": self.con_sigma,
                "说明": "贡献值差异惩罚项的标准差，控制贡献值差异对独立性检验的影响程度"
            },
            "high_contrib_fallback_percentile": {
                "值": self.high_contrib_fallback_percentile,
                "说明": "当无法找到密度拐点时，用于判定高贡献特征的分位数阈值"
            }
        }

    def print_parameter_summary(self):
        """打印参数摘要说明"""
        print("\n" + "=" * 50)
        print("因果图构建参数说明")
        print("=" * 50)

        explanations = self.get_parameter_explanations()

        # 分组打印参数
        print("\n【方法选择参数】")
        print("-" * 40)
        for param in ["high_contrib_method", "alpha_adjust_strategy", "stability_threshold_method"]:
            print(f"{param}:")
            print(f"  值: {explanations[param]['值']}")
            print(f"  说明: {explanations[param]['说明']}\n")

        print("\n【数值参数】")
        print("-" * 40)
        for param in ["max_alpha_reduction", "min_stability_threshold", "max_cond_reduction",
                      "fdr_control_level", "small_sample_threshold", "con_sigma",
                      "high_contrib_fallback_percentile"]:
            print(f"{param}:")
            print(f"  值: {explanations[param]['值']}")
            print(f"  说明: {explanations[param]['说明']}\n")

        print("=" * 50 + "\n")


# 全局配置和输出锁 - 使用multiprocessing.Lock确保跨进程安全
config = CausalConfig()
print_lock = Lock()  # 改为multiprocessing.Lock


# --------------------------
# 1. 并行工作函数
# --------------------------
def bootstrap_worker(args):
    boot_indices, features, is_numeric, target_name, current_alpha, max_path_length = args
    boot_features = [f.copy_with_data(f.data[boot_indices]) for f in features]

    if is_numeric:
        boot_corr = np.corrcoef([f.data for f in boot_features])
    else:
        boot_corr, _ = spearmanr([f.data for f in boot_features])

    boot_paths, _ = _build_with_alpha(
        boot_features, boot_corr, target_name, current_alpha,
        is_numeric, skip_prune=True, max_path_length=max_path_length
    )
    return [tuple(p) for p in boot_paths]


def process_alpha(args):
    alpha, features, corr_matrix, target_name, is_numeric, max_path_length = args
    # with print_lock:
    #     print(f"\n使用阈值alpha={alpha:.3f}构建因果图...")
    paths, dag = _build_with_alpha(
        features, corr_matrix, target_name, alpha, is_numeric,
        max_path_length=max_path_length
    )
    return alpha, paths, dag


class Feature:
    """封装节点（特征/目标变量）的核心属性"""

    def __init__(self, name: str, feature_names: list, all_con_values=None):
        self.name = name
        self.data = shared_globals.data[name].values  # 原始数据

        # 计算噪声（变异系数）
        try:
            self.data = np.asarray(self.data, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"特征 {name} 包含非数值数据: {e}") from e
        mean = np.mean(self.data)
        std = np.std(self.data)

        # 处理常量特征（标准差接近0）
        if std < 1e-9:
            self.noise = 0.0  # 常量特征噪声设为0
            with print_lock:
                print(f"特征 {name} 为常量特征，噪声强制设为0")
        else:
            # 增强分母保护，避免均值接近0导致的极端值
            denom = np.abs(mean) + 1e-6  # 增大保护值至1e-6，减少极端情况
            self.noise = std / denom

            # 对极端噪声值进行截断（可选，根据业务场景调整）
            if self.noise > 1e6:
                self.noise = 1e6
                with print_lock:
                    print(f"特征 {name} 噪声值过大，已截断至1e6")

        #self.noise = std / (np.abs(mean) + 1e-8)

        # 贡献值
        self.con_i = (shared_globals.con_list[feature_names.index(name)]
                      if name in feature_names else 0.0)
        self.all_con_values = all_con_values  # 存储传入的所有贡献值列表

        # 改进1：数据驱动的高贡献特征判定（基于密度拐点）
        self.is_high_contrib = False
        self.threshold = None  # 存储计算的阈值
        self.judgment_reason = ""  # 记录判定原因

        # 调试：打印贡献值和阈值
        if feature_names and name in feature_names and self.all_con_values is not None:
            # with print_lock:
            #     print(f"\n处理特征: {name}，贡献值: {self.con_i:.4f}")
            #     # print(f"所有特征贡献值: {[round(v, 4) for v in self.all_con_values]}")

            # 仅在有足够多样本时使用数据驱动方法
            if len(self.all_con_values) >= config.small_sample_threshold:
                self.is_high_contrib, self.threshold, self.judgment_reason = self._is_high_contrib_data_driven(self.all_con_values)
            else:
                # 小样本时使用相对保守的分位数方法
                threshold = np.percentile(self.all_con_values, 80)  # 小样本时更严格
                self.threshold = threshold
                self.is_high_contrib = self.con_i >= threshold
                self.judgment_reason = f"小样本（{len(self.all_con_values)} < {config.small_sample_threshold}），使用80%分位数阈值: {threshold:.4f}"

            # with print_lock:
            #     print(f"特征 {self.name} 判定结果: {'高贡献' if self.is_high_contrib else '非高贡献'}")
            #     print(f"判定依据: {self.judgment_reason}")
            #     print(f"特征贡献值: {self.con_i:.4f}, 计算阈值: {self.threshold:.4f}")

    def _is_high_contrib_data_driven(self, all_con_values):
        """基于贡献值分布的密度拐点确定高贡献特征，返回(是否高贡献, 阈值, 原因)"""
        # 核密度估计：自动选择带宽（使用silverman方法）
        kde = gaussian_kde(all_con_values, bw_method='silverman')  # 显式指定带宽方法
        sorted_con = np.sort(all_con_values)
        density = kde(sorted_con)

        # 平滑密度曲线，减少噪声影响
        from scipy.ndimage import gaussian_filter1d
        density_smoothed = gaussian_filter1d(density, sigma=1)  # 高斯平滑

        # 基于平滑后的密度计算导数
        first_deriv = np.gradient(density_smoothed)
        second_deriv = np.gradient(first_deriv)
        sign_changes = np.diff(np.sign(second_deriv))
        inflection_points = np.where(sign_changes != 0)[0]  # 二阶导数符号变化的点

        # with print_lock:
        #     print(f"二阶导数符号变化点索引: {inflection_points.tolist()}")
        #     print(
        #         f"拐点对应的贡献值: {[round(sorted_con[i], 4) if i < len(sorted_con) else None for i in inflection_points]}")

        # 选择最显著的拐点作为阈值
        if len(inflection_points) > 0:
            # 筛选在右侧区域的拐点（贡献值较高的部分）
            right_inflections = [i for i in inflection_points if i > len(sorted_con) * 0.3]
            if right_inflections:
                threshold_idx = right_inflections[-1]  # 取最右侧的拐点
                threshold = sorted_con[threshold_idx]
                max_con = max(all_con_values)
                adjusted_threshold = min(threshold, max_con * 0.9)  # 阈值不超过最大贡献值的90%

                reason = f"使用右侧拐点阈值: {threshold:.4f}，调整后: {adjusted_threshold:.4f}"
                return self.con_i >= adjusted_threshold, adjusted_threshold, reason

        # 没有合适的拐点，使用分位数
        threshold = np.percentile(all_con_values, config.high_contrib_fallback_percentile)
        reason = f"未找到有效拐点，使用{config.high_contrib_fallback_percentile}%分位数阈值: {threshold:.4f}"
        return self.con_i >= threshold, threshold, reason

    def copy_with_data(self, new_data):
        new_feature = Feature(self.name, [], None)
        new_feature.data = new_data
        new_feature.noise = self.noise
        new_feature.con_i = self.con_i
        new_feature.is_high_contrib = self.is_high_contrib
        new_feature.threshold = self.threshold
        new_feature.judgment_reason = self.judgment_reason
        return new_feature


def _find_paths(dag, target_idx, max_length=3):
    paths = []
    start_nodes = [i for i in range(dag.shape[0]) if i != target_idx and dag[i, target_idx] == 1]
    if not start_nodes and max_length == 1:
        return []

    from collections import deque
    queue = deque()
    for start in start_nodes:
        queue.append((start, [start], 1))

    while queue:
        current, path, length = queue.popleft()
        if current == target_idx:
            paths.append(path.copy())
            continue
        if length >= max_length:
            continue
        next_nodes = [j for j in range(dag.shape[0]) if j != current and dag[current, j] == 1 and j not in path]
        for next_node in next_nodes:
            new_path = path.copy()
            new_path.append(next_node)
            queue.append((next_node, new_path, length + 1))
    return paths


# 缓存骨架构建结果
@lru_cache(maxsize=10)
def cached_make_skeleton(n, sample_num, alpha, corr_matrix_tuple, feature_con, feature_noise,
                         is_high_contrib, contrib_values, enable_progress=True):
    adj = np.ones((n, n)) - np.eye(n)  # 邻接矩阵
    sep_sets = [[[] for _ in range(n)] for _ in range(n)]  # 分离集
    l = 0
    corr_matrix = np.array(corr_matrix_tuple)

    @lru_cache(maxsize=None)
    def cached_is_independent(i, j, cond_tuple):
        cond = list(cond_tuple)
        idx = [i, j] + cond
        corr_sub = corr_matrix[np.ix_(idx, idx)]
        corr_sub = np.nan_to_num(corr_sub, nan=0.0, posinf=1.0, neginf=-1.0)
        # 增强正则化，避免矩阵接近奇异
        eps = 1e-4  # 从1e-4增大到1e-3
        corr_sub_reg = corr_sub + eps * np.eye(corr_sub.shape[0])

        try:
            # 增加rcond参数，增强数值稳定性
            partial_corr = np.linalg.pinv(corr_sub_reg, rcond=1e-3)
        except np.linalg.LinAlgError:
            # 更鲁棒的 fallback
            partial_corr = np.eye(corr_sub_reg.shape[0]) * (1 / eps)

        denominator = np.sqrt(abs(partial_corr[0, 0] * partial_corr[1, 1]))
        rho = (-partial_corr[0, 1]) / denominator if denominator >= 1e-10 else 0.0
        rho = np.clip(rho, -0.99999, 0.99999)

        # 惩罚项计算
        con_i, con_j = feature_con[i], feature_con[j]
        noise_i, noise_j = feature_noise[i], feature_noise[j]
        con_penalty = np.exp(-(con_i - con_j) ** 2 / (2 * config.con_sigma ** 2))
        noise_penalty = np.exp(-0.5 * (noise_i + noise_j))
        sample_penalty = min(1.0, sample_num / 1000)
        omega = con_penalty * noise_penalty * sample_penalty

        t_stat = (rho / np.sqrt((1 - rho ** 2) / (sample_num - len(cond) - 2))) * omega
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))

        # 改进2：自适应alpha调整（而非固定的1/3）
        if is_high_contrib[i] or is_high_contrib[j]:
            # 贡献值越高，调整幅度越大（基于相对贡献值）
            max_con = max(contrib_values) if contrib_values else 1.0
            rel_con_i = con_i / max_con if max_con > 0 else 0
            rel_con_j = con_j / max_con if max_con > 0 else 0
            adjust_factor = 1 - (max(rel_con_i, rel_con_j) * config.max_alpha_reduction)
            adjusted_alpha = alpha * adjust_factor

            # with print_lock:
            #     print(
            #         f"特征对 ({i},{j}) 独立性检验: p值={p_value:.4f}, 原始alpha={alpha:.4f}, 调整后alpha={adjusted_alpha:.4f}")

            return p_value >= adjusted_alpha

        # with print_lock:
        #     print(f"特征对 ({i},{j}) 独立性检验: p值={p_value:.4f}, alpha={alpha:.4f}")

        return p_value >= alpha

    while True:
        updated = False
        pairs = combinations(range(n), 2)
        total_pairs = n * (n - 1) // 2

        # if enable_progress and threading.current_thread() is threading.main_thread():
        #     pairs = tqdm(pairs, desc=f"骨架构建（l={l}）", total=total_pairs, leave=False)

        for i, j in pairs:
            if adj[i, j] == 0:
                continue
            neighbors = [k for k in range(n) if adj[i, k] == 1 and k != j]
            if len(neighbors) >= l:
                # 改进3：基于贡献值的条件组合调整
                if is_high_contrib[i] or is_high_contrib[j]:
                    # 贡献越高，减少的条件组合越多（但不超过最大值）
                    max_con = max(contrib_values) if contrib_values else 1.0
                    rel_con = max(feature_con[i], feature_con[j]) / max_con if max_con > 0 else 0
                    cond_reduction = int(round(rel_con * config.max_cond_reduction))
                    max_cond = max(0, l - cond_reduction)

                    # with print_lock:
                    #     print(
                    #         f"高贡献特征对 ({i},{j}) 条件组合调整: 原始l={l}, 减少{cond_reduction}, 调整后max_cond={max_cond}")
                else:
                    max_cond = l

                for cond in combinations(neighbors, max_cond):
                    if cached_is_independent(i, j, cond):
                        adj[i, j] = adj[j, i] = 0
                        sep_sets[i][j] = sep_sets[j][i] = list(cond)
                        updated = True
                        break
        if not updated and l < n:
            l += 1
        else:
            break
    return (adj.tolist(), [sublist for sublist in sep_sets])


def _build_with_alpha(features, corr_matrix, target_name, alpha, is_numeric, skip_prune=False, max_path_length=3):
    sample_num = len(features[0].data)
    n = len(features)

    feature_con = tuple(f.con_i for f in features)
    feature_noise = tuple(f.noise for f in features)
    is_high_contrib = tuple(f.is_high_contrib for f in features)
    contrib_values = tuple(f.con_i for f in features if f.con_i > 0)  # 用于调整的贡献值
    corr_matrix_tuple = tuple(map(tuple, corr_matrix))

    # with print_lock:
    #     print(f"\n开始构建alpha={alpha:.3f}的因果图，样本数={sample_num}, 特征数={n}")
        # print(f"高贡献特征索引: {[i for i, val in enumerate(is_high_contrib) if val]}")

    enable_progress = threading.current_thread() is threading.main_thread()

    skeleton_list, sep_sets = cached_make_skeleton(
        n, sample_num, alpha, corr_matrix_tuple, feature_con, feature_noise,
        is_high_contrib, contrib_values, enable_progress
    )
    skeleton = np.array(skeleton_list)

    dag = extend_to_dag(skeleton, sep_sets, n)

    if skip_prune:
        target_idx = shared_globals.all_names.index(target_name[0])
        valid_paths = _find_paths(dag, target_idx, max_length=max_path_length)
    else:
        valid_paths = prune_paths(dag, features, shared_globals.all_names, target_name,
                                  alpha, is_numeric, max_path_length=max_path_length)

    # with print_lock:
    #     print(f"alpha={alpha:.3f} 构建完成，有效路径数={len(valid_paths)}")

    return valid_paths, dag


def build_causal_graph(features, corr_matrix, target_name, is_numeric,
                       alpha_list=None, max_path_length=None,):
    # 从config获取参数
    n_bootstrap = config.n_bootstrap
    con_percentile = config.con_percentile
    noise_percentile = config.noise_percentile
    min_valid_paths = config.min_valid_paths

    config.auto_tune_based_on_data(features)  # 新增：自动调整参数
    max_path_length = max_path_length or config.max_path_length  # 修改：使用配置中的值

    # 初始化best_alpha为默认值（使用最大的alpha）
    # alpha_list = alpha_list or [ 0.01, 0.03, 0.05,0.07,0.1]
    alpha_list = alpha_list or [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    best_alpha = max(alpha_list)  # 初始化默认值

    # 仅保留关键参数打印
    with print_lock:
        print("=" * 60)
        print("           因果图构建关键参数 summary")
        print("=" * 60)
        print(f"目标变量: {target_name}")
        print(f"特征总数: {len(features)}")
        print(
            f"使用的alpha阈值列表: {alpha_list or [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]}")
        print(f"最大路径长度限制: {max_path_length}")
        print(f"Bootstrap抽样次数: {n_bootstrap}")
        print(f"最小有效路径数阈值: {min_valid_paths}")
        print("-" * 60)

    shared_globals.target_idx = shared_globals.all_names.index(target_name[0])


    parallel_args = [
        (alpha, features, corr_matrix, target_name, is_numeric, max_path_length)
        for alpha in alpha_list
    ]

    with Pool(processes=min(os.cpu_count(), len(alpha_list))) as pool:
        results = list(tqdm(
            pool.imap(process_alpha, parallel_args),
            total=len(alpha_list), desc="阈值搜索进度"
        ))

    best_paths, best_dag = [], None
    for alpha, paths, dag in results:
        current_count = len(paths)
        with print_lock:
            print(f"alpha={alpha:.3f} 有效路径数: {current_count}")
        if current_count >= min_valid_paths and current_count > len(best_paths):
            best_paths, best_dag = paths, dag
            if current_count >= 10:
                with print_lock:
                    print("路径数充足，提前终止阈值搜索")
                break

    if not best_paths:
        with print_lock:
            print("未找到足够路径，使用最大alpha重试")
        best_paths, best_dag = _build_with_alpha(
            features, corr_matrix, target_name, max(alpha_list), is_numeric,
            max_path_length=max_path_length
        )

    if not best_paths:
        best_alpha = max(alpha_list)
        best_paths, best_dag = _build_with_alpha(
            features, corr_matrix, target_name, best_alpha, is_numeric,
            max_path_length=max_path_length
        )

    # 提取高贡献特征判定阈值
    high_contrib_thresholds = [f.threshold for f in features if f.is_high_contrib]
    overall_high_contrib_threshold = np.mean(high_contrib_thresholds) if high_contrib_thresholds else 0

    target_idx = shared_globals.target_idx
    path_feature_indices = set(idx for path in best_paths for idx in path if idx != target_idx)
    filtered_features = [shared_globals.all_names[i] for i in path_feature_indices]
    filtered_features.sort()
    shared_globals.filtered_features = filtered_features

    with print_lock:
        print(f"\n【有效路径涉及的特征】共{len(filtered_features)}个:")
        for i, feat in enumerate(filtered_features, 1):
            print(f"{i}. {feat}")

    draw_causal_graph(best_paths, features, shared_globals.all_names, target_name)
    classify_paths(best_paths, shared_globals.all_names, target_name)
    shared_globals.valid_paths = best_paths

    # 最终结果参数打印
    with print_lock:
        print("\n" + "=" * 60)
        print("           因果图构建最终结果")
        print("=" * 60)
        print(f"1. 最优alpha阈值: {best_alpha:.4f}")
        print(f"2. 有效路径总数: {len(best_paths)}")
        print(f"   - 直接路径数: {len([p for p in best_paths if len(p) - 1 == 1])}")
        print(f"   - 间接路径数: {len([p for p in best_paths if len(p) - 1 > 1])}")
        print(f"3. 高贡献特征判定阈值 (平均): {overall_high_contrib_threshold:.4f}")
        print(f"4. 高贡献特征数量: {sum(1 for f in features if f.is_high_contrib)}")
        print(f"5. 路径稳定性FDR控制水平: {config.fdr_control_level}")
        print(f"6. 最小稳定性阈值: {config.min_stability_threshold}")
        print("=" * 60 + "\n")

    # 替换原有的提取和打印逻辑
    high_contrib_features = get_and_print_high_contrib_features(features, print_lock)
    shared_globals.high_contrib_features = high_contrib_features  # 赋值全局变量

    return best_paths, best_dag, shared_globals.all_names


def _compute_independence(i, j, cond, corr_matrix, sample_num,
                          feature_con, feature_noise, is_high_contrib, alpha,
                          contrib_values, con_sigma=0.5):
    idx = [i, j] + cond
    corr_sub = corr_matrix[np.ix_(idx, idx)]
    corr_sub = np.nan_to_num(corr_sub, nan=0.0, posinf=1.0, neginf=-1.0)

    eps = 1e-4
    corr_sub_reg = corr_sub + eps * np.eye(corr_sub.shape[0])
    try:
        partial_corr = np.linalg.pinv(corr_sub_reg, rcond=1e-3)
    except np.linalg.LinAlgError:
        partial_corr = np.eye(corr_sub_reg.shape[0]) * (1 / eps)

    denominator = np.sqrt(abs(partial_corr[0, 0] * partial_corr[1, 1]))
    rho = (-partial_corr[0, 1] / denominator) if denominator >= 1e-10 else 0.0
    rho = np.clip(rho, -0.99999, 0.99999)

    con_i, con_j = feature_con[i], feature_con[j]
    noise_i, noise_j = feature_noise[i], feature_noise[j]
    con_penalty = np.exp(-(con_i - con_j) ** 2 / (2 * con_sigma ** 2))
    noise_penalty = np.exp(-0.5 * (noise_i + noise_j))
    sample_penalty = min(1.0, sample_num / 1000)
    omega = con_penalty * noise_penalty * sample_penalty

    t_stat = (rho / np.sqrt((1 - rho ** 2) / (sample_num - len(cond) - 2))) * omega
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    # 自适应alpha调整
    if is_high_contrib[i] or is_high_contrib[j]:
        max_con = max(contrib_values) if contrib_values else 1.0
        rel_con_i = con_i / max_con if max_con > 0 else 0
        rel_con_j = con_j / max_con if max_con > 0 else 0
        adjust_factor = 1 - (max(rel_con_i, rel_con_j) * config.max_alpha_reduction)
        adjusted_alpha = alpha * adjust_factor
        return p_value >= adjusted_alpha
    return p_value >= alpha


def is_independent(i, j, cond, features, corr_matrix, sample_num, alpha):
    feature_con = tuple(f.con_i for f in features)
    feature_noise = tuple(f.noise for f in features)
    is_high_contrib = tuple(f.is_high_contrib for f in features)
    contrib_values = tuple(f.con_i for f in features if f.con_i > 0)
    return _compute_independence(
        i, j, cond, corr_matrix, sample_num,
        feature_con, feature_noise, is_high_contrib, alpha, contrib_values
    )


def extend_to_dag(skeleton, sep_sets, n):
    dag = skeleton.copy()
    # 识别V-结构
    for i, j, k in combinations(range(n), 3):
        if dag[i, j] == 1 and dag[j, k] == 1 and dag[i, k] == 0:
            if j not in sep_sets[i][k]:
                dag[j, i] = dag[j, k] = 0  # 定向为i→j←k
    # Meek规则
    for i in range(n):
        for j in range(n):
            if i != j and dag[i, j] == 1 and dag[j, i] == 0:  # i→j
                for k in range(n):
                    if k != i and k != j and dag[j, k] == 1 and dag[k, j] == 1:  # j-k
                        if dag[i, k] == 0 and dag[k, i] == 0:  # i与k无连接
                            dag[k, j] = 0  # 定向为j→k
    return dag


from concurrent.futures import ThreadPoolExecutor, as_completed


def prune_paths(dag, features, all_names, target_name, current_alpha, is_numeric,
                max_path_length=3, n_bootstrap=30, con_percentile=30, noise_percentile=70):
    target_idx = all_names.index(target_name[0])
    paths = _find_paths(dag, target_idx, max_length=max_path_length)
    original_paths = [tuple(path) for path in paths]

    # with print_lock:
    #     print(f"\n路径剪枝开始，初始路径数: {len(original_paths)}")
    #     print(f"原始路径列表: {[[all_names[i] for i in path] for path in original_paths]}")

    if not original_paths:
        return []

    high_contrib_indices = [i for i, f in enumerate(features) if f.is_high_contrib]
    # with print_lock:
    #     high_contrib_names = [all_names[i] for i in high_contrib_indices]
    #     print(f"高贡献特征索引: {high_contrib_indices}, 名称: {high_contrib_names}")

    # Bootstrap抽样
    sample_size = len(features[0].data)
    boot_indices_list = [np.random.choice(sample_size, size=sample_size, replace=True)
                         for _ in range(n_bootstrap)]

    args_list = [
        (boot_indices, features, is_numeric, target_name, current_alpha, max_path_length)
        for boot_indices in boot_indices_list
    ]

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), n_bootstrap)) as executor:
        with print_lock:
            progress_bar = tqdm(total=n_bootstrap, desc="Bootstrap抽样验证")
        boot_results = []
        futures = [executor.submit(bootstrap_worker, args) for args in args_list]
        for future in as_completed(futures):
            boot_results.append(future.result())
            with print_lock:
                progress_bar.update(1)
        with print_lock:
            progress_bar.close()

    # 统计路径频率
    path_freq = defaultdict(int)
    for res in boot_results:
        for p in res:
            if p in original_paths:
                path_freq[p] += 1

    # with print_lock:
    #     print("\nBootstrap路径频率统计:")
    #     for path, freq in path_freq.items():
    #         path_names = [all_names[i] for i in path]
    #         print(f"路径 {path_names}: 出现频率 {freq / n_bootstrap:.2f} ({freq}/{n_bootstrap})")

    # 改进4：基于FDR的稳定性阈值（而非固定的50%/30%）
    stable_paths = []
    if original_paths:
        # 计算所有路径的频率
        frequencies = np.array([path_freq.get(p, 0) / n_bootstrap for p in original_paths])
        path_indices = np.arange(len(original_paths))

        # with print_lock:
        #     print(f"所有路径频率: {[round(f, 4) for f in frequencies]}")

        # 按频率排序并计算FDR
        sorted_indices = np.argsort(frequencies)[::-1]
        sorted_freq = frequencies[sorted_indices]

        # 错误发现率计算（Benjamini-Hochberg方法）
        fdr_values = (sorted_freq * len(frequencies)) / (np.arange(len(frequencies)) + 1)

        # with print_lock:
        #     print(f"排序后的路径频率: {[round(f, 4) for f in sorted_freq]}")
        #     print(f"计算的FDR值: {[round(f, 4) for f in fdr_values]}")
        #     print(f"FDR控制水平: {config.fdr_control_level}")

        significant_mask = fdr_values <= config.fdr_control_level

        if np.any(significant_mask):
            # 找到最大显著频率阈值
            min_significant_freq = sorted_freq[significant_mask][-1]
            # 确保阈值不低于最小值
            min_significant_freq = max(min_significant_freq, config.min_stability_threshold)

            # with print_lock:
            #     print(f"显著路径阈值: {min_significant_freq:.4f}")

            # 应用阈值
            for i, path in enumerate(original_paths):
                freq = path_freq.get(path, 0) / n_bootstrap
                path_names = [all_names[i] for i in path]
                has_high_contrib = any(idx in high_contrib_indices for idx in path
                                       if idx != target_idx)

                # 高贡献路径可适当降低阈值（但不低于最小显著阈值）
                adjusted_threshold = max(min_significant_freq * 0.8, config.min_stability_threshold)

                if freq >= min_significant_freq or (has_high_contrib and freq >= adjusted_threshold):
                    stable_paths.append(list(path))
                    # with print_lock:
                    #     print(
                    #         f"路径 {path_names} 保留: 频率 {freq:.4f} {'(高贡献路径放宽阈值)' if has_high_contrib else ''}")
                # else:
                #     with print_lock:
                #         print(
                #             f"路径 {path_names} 剔除: 频率 {freq:.4f} < {'调整后阈值' if has_high_contrib else '阈值'} {adjusted_threshold if has_high_contrib else min_significant_freq:.4f}")
        else:
            # 当没有显著路径时，使用最小阈值
            # with print_lock:
            #     print(f"未发现显著稳定的路径，使用最小稳定性阈值 {config.min_stability_threshold}")
            for path in original_paths:
                freq = path_freq.get(path, 0) / n_bootstrap
                path_names = [all_names[i] for i in path]
                if freq >= config.min_stability_threshold:
                    stable_paths.append(list(path))
                    # with print_lock:
                    #     print(f"路径 {path_names} 保留: 频率 {freq:.4f} >= 最小阈值 {config.min_stability_threshold}")
                # else:
                #     with print_lock:
                #         print(f"路径 {path_names} 剔除: 频率 {freq:.4f} < 最小阈值 {config.min_stability_threshold}")

    # 特征质量筛选
    all_con = [f.con_i for f in features if f.name != target_name[0]]
    all_noise = [f.noise for f in features if f.name != target_name[0]]
    if not all_con or not all_noise:
        return []

    con_threshold = np.percentile(all_con, con_percentile)
    noise_threshold = np.percentile(all_noise, noise_percentile)

    # with print_lock:
    #     print(f"\n特征质量筛选阈值: 贡献值={con_threshold:.4f}, 噪声={noise_threshold:.4f}")

    valid = []
    for path in stable_paths:
        if path[-1] != target_idx or not (1 <= len(path) - 1 <= max_path_length):
            # with print_lock:
            #     print(f"路径 {[all_names[i] for i in path]} 剔除: 长度不符合要求")
            continue

        feature_indices = [idx for idx in path if idx != target_idx]
        if not feature_indices:
            # with print_lock:
            #     print(f"路径 {[all_names[i] for i in path]} 剔除: 无有效特征")
            continue

        mean_con = np.mean([features[idx].con_i for idx in feature_indices])
        has_high_contrib = any(idx in high_contrib_indices for idx in feature_indices)
        path_names = [all_names[i] for i in path]

        # 改进5：基于贡献分布的动态阈值调整
        if has_high_contrib:
            # 根据贡献值分布形状调整阈值
            con_distribution = np.array(all_con)
            con_iqr = np.percentile(con_distribution, 75) - np.percentile(con_distribution, 25)
            # 分布越分散，阈值调整幅度越大
            adjustment = 0.5 + (con_iqr / (np.mean(con_distribution) + 1e-8)) * 0.2
            adjusted_con_threshold = con_threshold * min(0.8, adjustment)

            # with print_lock:
            #     print(
            #         f"高贡献路径 {path_names} 贡献值检查: 均值={mean_con:.4f}, 调整后阈值={adjusted_con_threshold:.4f}")

            # if mean_con < adjusted_con_threshold:
            #     with print_lock:
            #         print(f"路径 {path_names} 剔除: 贡献值均值不足")
            #     continue
        # else:
            # with print_lock:
            #     print(f"路径 {path_names} 贡献值检查: 均值={mean_con:.4f}, 阈值={con_threshold:.4f}")
            #
            # if mean_con < con_threshold:
            #     with print_lock:
            #         print(f"路径 {path_names} 剔除: 贡献值均值不足")
            #     continue

        # 噪声阈值调整
        middle_noise = np.mean([features[i].noise for i in path[1:-1]]) if len(path) >= 3 else 0
        if has_high_contrib:
            noise_distribution = np.array(all_noise)
            noise_iqr = np.percentile(noise_distribution, 75) - np.percentile(noise_distribution, 25)
            adjustment = 1.2 + (noise_iqr / (np.mean(noise_distribution) + 1e-8)) * 0.3
            adjusted_noise_threshold = noise_threshold * min(1.5, adjustment)

            # with print_lock:
            #     print(
            #         f"高贡献路径 {path_names} 噪声检查: 噪声={middle_noise:.4f}, 调整后阈值={adjusted_noise_threshold:.4f}")

            if middle_noise <= adjusted_noise_threshold:
                valid.append(path)
                # with print_lock:
                #     print(f"[高贡献路径保留] {path_names} (噪声: {middle_noise:.4f})")
            # else:
            #     with print_lock:
            #         print(f"路径 {path_names} 剔除: 噪声过高")
        else:
            # with print_lock:
            #     print(f"路径 {path_names} 噪声检查: 噪声={middle_noise:.4f}, 阈值={noise_threshold:.4f}")

            if middle_noise <= noise_threshold:
                valid.append(path)
                # with print_lock:
                #     print(f"[路径保留] {path_names} (噪声: {middle_noise:.4f})")
            # else:
            #     with print_lock:
            #         print(f"路径 {path_names} 剔除: 噪声过高")

    return valid


def draw_causal_graph(valid_paths, features, all_names, target_name):
    G = nx.DiGraph()
    target_node = target_name[0]
    G.add_nodes_from(all_names)

    for path in valid_paths:
        for i in range(len(path) - 1):
            G.add_edge(all_names[path[i]], all_names[path[i + 1]])

    pos = {target_node: (0, 0)}
    others = [n for n in all_names if n != target_node]
    angle = 2 * np.pi / len(others) if others else 0
    pos.update({others[i]: (2 * np.cos(i * angle), 2 * np.sin(i * angle))
                for i in range(len(others))})

    all_con = [f.con_i for f in features]
    max_con = max(all_con) if all_con and max(all_con) != 0 else 1.0
    node_size = [f.con_i / max_con * 2000 for f in features]
    node_colors = ['lightgreen' if f.is_high_contrib else 'lightblue' for f in features]

    edges = G.edges()
    edge_colors = [(features[all_names.index(u)].noise + features[all_names.index(v)].noise) / 2
                   for u, v in edges]
    edge_colors = [plt.cm.RdYlGn(1 - min(n, 0.6) / 0.6) for n in edge_colors]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                           arrowstyle='->', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos,
                            labels={n: f"{n}\n{features[i].con_i:.5f}"
                                    for i, n in enumerate(all_names)},
                            font_size=9)

    plt.title("因果特征图（绿色节点=高贡献特征，节点大小=贡献值，边颜色=噪声）")
    #save_filename = f"{dataset_name}_{model_name}.keras"
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def classify_paths(valid_paths, all_names, target_name):
    direct, indirect = [], []
    for path in valid_paths:
        path_str = " → ".join([all_names[i] for i in path])
        if len(path) - 1 == 1:
            direct.append(path_str)
        else:
            indirect.append(path_str)

    shared_globals.direct_paths = direct
    shared_globals.indirect_paths = indirect

    with print_lock:
        print("\n【直接路径（特征→target）】")
        for i, p in enumerate(direct, 1):
            print(f"{i}. {p}")
        print("\n【间接路径（特征→...→target）】")
        for i, p in enumerate(indirect, 1):
            print(f"{i}. {p}")


if __name__ == '__main__':
    if not hasattr(shared_globals, 'feature_max_abs'):
        raise ValueError("请先运行特征扰动模块")

    # 打印参数说明
    # config.print_parameter_summary()

    # 预计算所有贡献值，用于数据驱动的高贡献判定
    all_con_values = [shared_globals.con_list[i] for i in range(len(shared_globals.feature_names))]
    with print_lock:
        print(
            f"\n所有特征贡献值列表: {[(shared_globals.feature_names[i], round(all_con_values[i], 4)) for i in range(len(all_con_values))]}")
        print(f"贡献值样本量: {len(all_con_values)}, 小样本阈值: {config.small_sample_threshold}")

    # 传递全局贡献值列表用于高贡献特征判定
    features = [Feature(name=n, feature_names=shared_globals.feature_names, all_con_values=all_con_values)
                for n in shared_globals.all_names]

    is_numeric = all(f.data.dtype.kind in 'iufc' for f in features)
    with print_lock:
        print(f"\n数据类型检查: {'数值型' if is_numeric else '非数值型'}")

    if is_numeric:
        corr_matrix = np.corrcoef([f.data for f in features])
    else:
        corr_matrix, _ = spearmanr([f.data for f in features])

    valid_paths, dag, _ = build_causal_graph(features, corr_matrix,
                                             shared_globals.target_name,
                                             is_numeric)
