#解决因果图构建时的并行计算问题
import tensorflow as tf
import logging

# 设置TensorFlow日志级别为WARNING（屏蔽INFO及以下级别的日志）
tf.get_logger().setLevel(logging.ERROR)
import os
import time

import numpy as np
# import tensorflow.python.keras.models
# from keras.models import load_model
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
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 1. 初始化全局变量
    shared_globals.init()
    # # 文件路径处理（兼容不同操作系统）
    data_path = os.path.join("dataset", "MagicGamma.csv")
    model_name = "CNN"

    # 2. 训练模型（新增步骤）
    print("===== Start model training...(Origin) =====")
    shared_globals.model=train(data_path, shared_globals.filtered_features)
    # 传入数据集路径和模型名称（模型名称可自定义，如"1d_cnn"）
    print("\nsaving keras model...(Origin) ")
    save_model(shared_globals.model, data_path, model_name)
    # print("模型预期输入形状:", shared_globals.model.input_shape)
    original_feature_len=shared_globals.feature_data.shape[1]

    # 记录各模块用时的变量
    feature_perturb_time = 0
    causal_graph_time = 0
    feature_recover_time = 0


    # 2. 运行特征扰动模块
    print("===== CauFeature beginning =====")
    print("===== 1.Run the Feature Perturbation module =====")
    start_time = time.time()
    perturb = CausalFeaturePerturbation(shared_globals.model, shared_globals.data)
    perturb.run_perturbation()  # 结果自动写入全局变量
    perturb.initialize_original_preds(shared_globals.data, shared_globals.model)

    # 打印每个特征的最大绝对差异
    print("List of Feature Contributions/Maximum JS Divergence (sorted by degree of influence)：")
    for idx, (col, val) in enumerate(shared_globals.feature_max_abs):
        print(f"  feature {col}: {round(val, 4)}")
    print("=" * 50)

    # 验证是否存入成功
    print("Predicted values of real samples:", shared_globals.original_preds)
    shared_globals.feature_max_abs.sort(key=lambda x: x[1], reverse=True)

    # 特征扰动模块计时结束
    feature_perturb_time = time.time() - start_time
    # print(f"特征扰动模块用时: {feature_perturb_time:.4f}秒")

    # 3. 运行因果图构建模块
    print("\n===== 2.Run the Causal Graph Construction module =====")
    start_time = time.time()


    # 实例化对象
    features = [Feature(name=n, feature_names=shared_globals.feature_names,all_con_values=shared_globals.con_list) for n in shared_globals.all_names]

    is_numeric = all(f.data.dtype.kind in 'iufc' for f in features)

    # 4. 计算相关系数矩阵（只计算一次）
    if is_numeric:
        shared_globals.corr_matrix = np.corrcoef([f.data for f in features])
    else:
        shared_globals.corr_matrix, _ = spearmanr([f.data for f in features])

    valid_paths, dag, _ = build_causal_graph(features, shared_globals.corr_matrix, shared_globals.target_name,is_numeric)


    # 输出结果
    print("\nDirected Graph Matrix：")
    print(f"Node Order：{shared_globals.all_names}")
    print(dag)

    # 因果图构建模块计时结束
    causal_graph_time = time.time() - start_time
    # print(f"因果图构建模块用时: {causal_graph_time:.4f}秒")

    print("\n===== 3.Run the Synergistic Interactive Features Incorporation module =====")
    start_time = time.time()

    shapley_module = PathShapleyModule(log_verbose=True, decay_type="exponential", alpha=0.7)

    shapley_module.run_redundant_check()

    # 特征恢复模块计时结束
    feature_recover_time = time.time() - start_time
    # print(f"特征恢复模块用时: {feature_recover_time:.4f}秒")



    # 4. 路径Shapley计算：直接读取全局分类路径
    # print("\n===== 运行路径Shapley值计算模块 =====")
    # result = shapley_module.run()
    #
    # # 输出结果
    # print("\n=== 特征级效应概率 ===")
    #
    # for feat, prob in result['feature_probabilities'].items():
    #     print(
    #         f"{feat}: 直接效应={prob['direct_effect']:.6f}（{prob['p_direct']:.2%}）, "
    #         f"间接效应={prob['indirect_effect']:.6f}（{prob['p_indirect']:.2%}）, "
    #         f"总效应={prob['direct_effect']+prob['indirect_effect']:.6f}"
    #     )
    #
    # print(f"\n总因果效应：{result['total_causal_effect']:.6f}")
    # print(f"概率和验证：{'通过' if result['probability_sum_valid'] else '失败'}")


    # 5. 特征选择后模型测试
    shared_globals.model=None
    filtered_features = [f for f in shared_globals.feature_names if f in shared_globals.filtered_features]
    #  训练模型（新增步骤）
    print("===== Start model training...(after CauFeature) =====")
    train(data_path, shared_globals.filtered_features)
    print("\nsaving keras model...(CauFeature) ")

    print(f"original dimension:{original_feature_len}")
    print(f"CauFeature dimension:{len(shared_globals.filtered_features) if shared_globals.filtered_features is not None else 0}")

    print(f"The model's execution time has been reduced by {shared_globals.running_time:.2f}s")
    print(f"The model's accuracy has been increased  by{shared_globals.accuracy:.2f}%")
    # print(f"The model's F-measure has been increased  by{shared_globals.f_score:.2f}%")

    # if shared_globals.accuracy >= 0:
    #     print(f"模型准确率提升了{shared_globals.accuracy:.2f}%")
    # else:
    #     print(f"模型准确率降低了{(0 - shared_globals.accuracy):.2f}%")

    # # 输出各模块用时汇总
    print("\n===== Each module's execution time =====")
    print(f"Feature Perturbation: {feature_perturb_time:.4f}s")
    print(f"Causal Graph Construction: {causal_graph_time:.4f}s")
    print(f"Key Feature Recovery: {feature_recover_time:.4f}s")
    print(f"Total Time: {feature_perturb_time + causal_graph_time + feature_recover_time:.4f}s")





