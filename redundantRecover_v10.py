import os

import shared_globals
import logging
from tqdm import tqdm
from typing import  Dict

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tqdm

class PathShapleyModule:
    """路径Shapley值计算模块（添加路径长度衰减功能）"""

    def __init__(self, log_verbose: bool = False, decay_type: str = "inverse", alpha: float = 0.8):
        self.decay_type = decay_type
        self.alpha = alpha
        self.log_verbose = log_verbose
        self._check_required_globals()

        # 复用全局变量，减少重复定义
        self.model = shared_globals.model
        self.feature_names = shared_globals.feature_names
        self.target_name = shared_globals.all_names[-1]
        self.feature_data = shared_globals.feature_data  # 不含目标列的特征数据
        self.original_preds = shared_globals.original_preds
        self.path_separators = shared_globals.path_separators

        # 节点传导效率（复用特征扰动阶段的贡献值和噪声）
        self.node_efficiency = self._compute_node_efficiency()

        # 初始化基础参数
        self.input_dim = len(self.feature_names)
        self.baseline = shared_globals.baselines
        self.sample_size = self.feature_data.shape[0] if not self.feature_data.empty else 0
        self.max_sample = getattr(shared_globals, 'max_sample', 1000)
        self.cache = {}  # 缓存路径子集结果
        self.baseline_pred = self._init_baseline_pred()  # 提前初始化基线预测值


        # 日志配置
        self.logger = self._init_logger()
        decay_desc = f"1/L（L为路径长度）" if decay_type == "inverse" else f"α^(L-1)（α={alpha}）"
        #self.logger.info(f"路径衰减模式：{decay_desc}")

        self._validate_model_input_shape()

    def _init_logger(self) -> logging.Logger:
        """初始化日志（复用逻辑抽取为函数）"""
        logger = logging.getLogger("PathShapley")
        logger.setLevel(logging.INFO if self.log_verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger

    def _init_baseline_pred(self) -> float:
        """初始化基线预测值（修复类型转换）"""
        samples = self.feature_data
        input_shape = self.model.input_shape[1:]  # 获取模型输入维度（排除batch）
        reshaped = samples.values.reshape(-1, *input_shape)  # 动态匹配形状
        # 转换为NumPy数组后展平
        baseline_preds = self.model.predict(reshaped, verbose=0).flatten()
        return baseline_preds.mean()

    def _validate_model_input_shape(self) -> None:
        if not hasattr(self.model, 'input_shape'):
            if self.log_verbose:
                self.logger.warning("模型无input_shape属性，跳过形状验证")
            return
        model_input_dim = self.model.input_shape[1] if len(self.model.input_shape) == 3 else self.model.input_shape[0]
        if self.input_dim != model_input_dim:
            raise ValueError(f"特征数量与模型输入不匹配：特征数={self.input_dim}，模型期望={model_input_dim}")


    def run_redundant_check(self):
        """
        整合冗余特征识别、组合评估和验证的完整流程
        """
        # 1. 获取因果图剔除的冗余特征
        self.get_redundant_features()

        #
        # print(f"识别到 {len(shared_globals.redundant_features)} 个冗余特征：{shared_globals.redundant_features}")
        # if not shared_globals.redundant_features:
        #     return

        print(f"识别到 {len(shared_globals.redundant_features)} 个非因果特征：{shared_globals.redundant_features}")
        print(f"识别到 {len(shared_globals.high_contrib_features)} 个高贡献特征：{shared_globals.high_contrib_features}")

        # 新增：执行双特征扰动
        self.run_double_feature_perturbation(threshold=1.0)  # 可调整阈值


    def get_redundant_features(self):
        """
        从全局特征中减去有效路径涉及的特征，得到冗余特征
        """
        # 将列表转换为集合进行差集运算，再转换回列表（保持顺序的话可用列表推导式）
        shared_globals.redundant_features = list(
            set(shared_globals.feature_names) - set(shared_globals.filtered_features if shared_globals.filtered_features is not None else []))
        return shared_globals.redundant_features

    def run_double_feature_perturbation(self, threshold: float) -> None:
        """
        执行双特征扰动：验证因果子集与非因果子集的交互效应，扩展因果子集
        :param threshold: 联合效应大于单个之和的阈值（如1.2表示联合效应需超过1.2倍单个之和）
        """
        # 1. 获取因果子集和非因果子集
        causal_features = shared_globals.filtered_features

        redundant_set = set(shared_globals.redundant_features)
        high_contrib_set = set(shared_globals.high_contrib_features)

        intersection = redundant_set & high_contrib_set

        non_causal_features =  list(intersection)

        print(f"非因果特征子集∩高贡献特征子集为：{non_causal_features}")

        # if not causal_features:
        #     self.logger.warning("因果子集为空，无法执行双特征扰动")
        #     return
        if not non_causal_features:
            self.logger.info("非因果特征子集∩高贡献特征子集为空，无需执行双特征扰动")
            return

        self.logger.info(f"开始双特征扰动：因果特征{len(causal_features)}个，非因果特征∩高贡献特征{len(non_causal_features)}个")

        # 2. 初始化依赖组件（特征扰动实例和单特征效应映射）
        if not hasattr(shared_globals, 'perturbation_instance'):
            raise ValueError("未初始化特征扰动实例，请先运行CausalFeaturePerturbation")
        perturbation = shared_globals.perturbation_instance  # 假设全局存储了扰动实例
        single_effect = self._get_single_feature_effect()

        # 3. 生成因果子集与非因果子集的笛卡尔积（双特征组合）
        # from itertools import product
        # feature_pairs = list(product(causal_features, non_causal_features))
        # self.logger.info(f"生成双特征组合共{len(feature_pairs)}对")
        # 新增：基于贡献值的启发式组合生成
        feature_pairs = self._heuristic_pair_selection(causal_features, non_causal_features)
        self.logger.info(f"生成双特征组合共{len(feature_pairs)}对")

        # 4. 并行遍历组合，评估联合效应
        to_add = set()
        to_add_lock = threading.Lock()  # 线程安全锁，保护to_add的修改
        max_workers = min(os.cpu_count(), len(feature_pairs))  # 限制最大线程数（根据CPU核心数调整）

        # 定义单个特征对的处理函数（供并行调用）
        def process_pair(c_feat, nc_feat):
            # 跳过已标记的非因果特征（减少重复计算）
            with to_add_lock:
                if nc_feat in to_add:
                    return None

            # 计算联合效应
            try:
                print(f"\n----- 开始计算组合 ({c_feat}, {nc_feat}) 的联合效应 -----")  # 新增
                joint_effect = perturbation.calculate_joint_contributions([c_feat, nc_feat])
            except Exception as e:
                # 新增：输出失败组合的详细信息
                print(f"\n===== 组合 ({c_feat}, {nc_feat}) 计算失败 =====")
                print(f"错误原因: {str(e)}")
                # 打印特征的扰动值状态
                for feat in [c_feat, nc_feat]:
                    if feat in shared_globals.feature_names:
                        idx = shared_globals.feature_names.index(feat)
                        perturb_vals = shared_globals.x_list[idx]
                        print(f"特征 {feat} 的扰动值长度: {len(perturb_vals)}")
                        print(f"特征 {feat} 的前3个扰动值: {perturb_vals[:3] if len(perturb_vals) >= 3 else []}")
                self.logger.warning(f"计算组合({c_feat}, {nc_feat})联合效应失败：{e}")
                return None

            # 获取特征对的节点效率
            c_eff = self.node_efficiency.get(c_feat, 0.0)  # 因果特征效率
            nc_eff = self.node_efficiency.get(nc_feat, 0.0)  # 非因果特征效率
            max_eff = max(self.node_efficiency.values()) if self.node_efficiency else 1.0
            mean_eff_ratio = (c_eff + nc_eff) / (2 * max_eff)  # 平均相对效率（0~1）

            # 动态阈值：效率越高，阈值越低（最低0.8倍基础阈值）
            dynamic_threshold = threshold * (1 - 0.4 * mean_eff_ratio)

            # 计算单特征效应之和
            c_effect = single_effect.get(c_feat, 0.0)
            nc_effect = single_effect.get(nc_feat, 0.0)
            sum_single = c_effect + nc_effect

            # 判断是否需要添加该非因果特征
            if sum_single < 1e-9:
                if joint_effect > 1e-6:
                    self.logger.info(f"组合({c_feat}, {nc_feat})：联合效应{joint_effect:.4f} > 单特征和0，标记为显著")
                    return nc_feat
            else:
                if joint_effect > sum_single * dynamic_threshold:
                    self.logger.info(
                        f"组合({c_feat}, {nc_feat})：联合效应{joint_effect:.4f} > 单特征和{sum_single:.4f}×{dynamic_threshold}，标记为显著"
                    )
                    return nc_feat
            return None

        # 提交并行任务并处理结果
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有特征对任务
            futures = {
                executor.submit(process_pair, c_feat, nc_feat): (c_feat, nc_feat)
                for c_feat, nc_feat in feature_pairs
            }

            # 跟踪进度并收集结果
            for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="双特征扰动评估（并行）"
            ):
                result = future.result()
                if result is not None:  # 若返回非None，说明需要添加该特征
                    with to_add_lock:
                        to_add.add(result)

        # 5. 更新因果子集
        if to_add:
            extended_causal = list(set(causal_features) | to_add)
            shared_globals.filtered_features = extended_causal
            self.logger.info(
                f"双特征扰动完成，新增{len(to_add)}个特征到特征选择子集：{sorted(to_add)}，"
                f"更新后特征选择子集共{len(extended_causal)}个特征"
            )
        else:
            self.logger.info("未发现显著交互效应的非因果-高贡献特征，因果子集保持不变")

    def _heuristic_pair_selection(self, causal_features, non_causal_features, top_k=200):
        """基于贡献值选择高潜力特征对"""
        # 获取贡献值
        causal_contrib = {
            f: shared_globals.con_list[shared_globals.feature_names.index(f)]
            for f in causal_features
        }
        non_causal_contrib = {
            f: shared_globals.con_list[shared_globals.feature_names.index(f)]
            for f in non_causal_features
        }

        # 计算组合分数并排序
        pairs = []
        for c_feat in causal_features:
            for nc_feat in non_causal_features:
                # 组合分数：贡献值乘积（优先高贡献组合）
                score = causal_contrib[c_feat] * non_causal_contrib[nc_feat]
                pairs.append((-score, c_feat, nc_feat))  # 负号用于升序排序

        # 选择分数最高的前k对
        pairs.sort()
        selected = [(c, nc) for (_, c, nc) in pairs[:top_k]]
        return selected

    def _get_single_feature_effect(self) -> Dict[str, float]:
        """生成特征名到单特征效应值的映射（复用con_list）"""
        if not hasattr(shared_globals, 'con_list') or not hasattr(shared_globals, 'feature_names'):
            raise ValueError("缺失特征贡献值数据，请先运行特征扰动模块")
        return {
            feat: shared_globals.con_list[i]
            for i, feat in enumerate(shared_globals.feature_names)
        }


    def _check_required_globals(self) -> None:
        required = [
            'model', 'feature_names', 'baselines', 'all_names',
            'path_separators', 'tolerance', 'random_seed',
            'feature_data', 'original_preds'
        ]
        missing = [var for var in required if not hasattr(shared_globals, var)]
        if missing:
            raise ValueError(f"全局变量缺失，无法初始化模块：{missing}")


    def _compute_node_efficiency(self) -> Dict[str, float]:
        """计算节点效率（复用全局变量）"""
        epsilon = 1e-10
        efficiency = {feat: shared_globals.con_list[i] / (shared_globals.feature_noise.get(feat, 0.0) + epsilon)
                      for i, feat in enumerate(self.feature_names)}
        efficiency[self.target_name] = 1.0
        return efficiency

