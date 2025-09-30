import random
import time
import copy
import scipy.stats as stats
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，仅保存图形不显示
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from sklearn.metrics import classification_report

# 关键：主进程提前初始化GPU（避免子进程重复初始化）
def init_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 主进程统一设置GPU内存动态增长，子进程不再修改
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"主进程：GPU初始化成功，逻辑GPU数量：{len(logical_gpus)}", flush=True)
        except RuntimeError as e:
            print(f"主进程：GPU初始化失败：{e}", flush=True)


# 关键：设置多进程启动方式为spawn（必须在创建进程池前执行）
def init_multiprocessing():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 若已设置过，忽略异常


import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from scipy.stats import spearmanr
import pandas as pd


def set_feature_dict(column_names, feature_dict):  # Assigning quantitative information
    for feature in column_names:
        feature_dict[feature] = {'change_range': 0, 'correlation': False, 'target': 0}
    return feature_dict


def Feature_correlation_analysis(column_names, feature_dict, spearmanr_data):  # Correlation between features
    Feature_correlation_list = []
    correlation_matrix, _ = spearmanr(spearmanr_data, axis=0)
    threshold = 0.7
    correlation_matrix[np.abs(correlation_matrix) < threshold] = np.nan
    plt.figure(figsize=(10, 8))
    mask = np.tri(*correlation_matrix.shape, k=-1)
    correlation_matrix = np.ma.array(correlation_matrix, mask=mask)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.xticks(range(len(column_names)), column_names, rotation=45, horizontalalignment='right')
    plt.yticks(range(len(column_names)), column_names)
    plt.title("Spearman")
    plt.close()  # 修复：保存前关闭图形，避免内存泄漏
    rela_class = []
    count = -1
    rows, cols = np.where(~np.isnan(correlation_matrix))
    for row, col in zip(rows, cols):
        if row != col and correlation_matrix[row, col] != 'masked':
            Feature_correlation_list.append(
                str(f"Feature {column_names[row]} and Feature {column_names[col]} have correlation: {correlation_matrix[row, col]:.2f}"))
            feature1 = column_names[row]
            feature2 = column_names[col]
            if not rela_class:
                rela_class.append([feature1, feature2])
                continue
            flag = 0
            count = -1
            for i in rela_class:
                count += 1
                if feature1 in i and feature2 not in i:
                    rela_class[count].append(feature2)
                    flag = 1
                    break
                elif feature2 in i and feature1 not in i:
                    rela_class[count].append(feature1)
                    flag = 1
                    break
                elif feature1 in i and feature2 in i:
                    flag = 1
                    break
            if flag == 0:
                rela_class.append([feature1, feature2])
    # 合并重叠的关联类
    result = []
    for sublist in rela_class:
        merged = False
        for existing in result:
            if any(item in existing for item in sublist):
                existing.extend(item for item in sublist if item not in existing)
                merged = True
                break
        if not merged:
            result.append(sublist)
    rela_class = result
    # 更新feature_dict的相关性标记
    for i in rela_class:
        for j in i:
            if j in feature_dict:
                feature_dict[j]['correlation'] = True
    # 计算每个特征的相关特征数量（NOFR）
    for feature in column_names:
        if feature_dict[feature]['correlation'] is False:
            feature_dict[feature]['NOFR'] = 0
        else:
            for corr_group in rela_class:
                if feature in corr_group:
                    feature_dict[feature]['NOFR'] = len(corr_group) - 1
                    break
    return rela_class, feature_dict, Feature_correlation_list


def target_correlation_analysis(feature_dict, target_variable, column_names, encoded_df):  # 新增encoded_df参数
    target_variable_list = []
    unique_values = np.unique(target_variable.values)
    if len(unique_values) != 2:
        # 多分类：用Spearman相关（使用编码后的特征）
        for feature in column_names:
            corr, _ = spearmanr(target_variable, encoded_df[feature])  # 替换df为encoded_df
            feature_dict[feature]['target'] = corr
            target_variable_list.append(str(f"{feature} - target: {corr:.2f}"))
    else:
        # 二分类：用点二列相关（使用编码后的特征）
        for feature in column_names:
            corr, _ = stats.pointbiserialr(encoded_df[feature], target_variable)  # 替换df为encoded_df
            feature_dict[feature]['target'] = corr
            target_variable_list.append(str(f"{feature} - target: {corr:.2f}"))
    return feature_dict, target_variable_list


def get_col_ranges(data, col_name):  # Analysing the values of features
    col_data = data[col_name]
    min_val = col_data.min()
    max_val = col_data.max()
    unique_count = col_data.nunique()
    unique_vals = sorted(col_data.unique().tolist())
    return min_val, max_val, unique_count, unique_vals


def value_range(column_names, df):  # Getting the feature statistics
    min_list, max_list, count_list, range_list = [], [], [], []
    for feat in column_names:
        min_val, max_val, unique_count, unique_vals = get_col_ranges(df, feat)
        min_list.append(min_val)
        max_list.append(max_val)
        count_list.append(unique_count)
        range_list.append(unique_vals)
    return min_list, max_list, count_list, range_list


def Disturbance_range(column_names, min_list, max_list, count_list, range_list):  # Feature Perturbation
    x_list = []
    for j in range(len(column_names)):
        min_val = min_list[j]
        max_val = max_list[j]
        unique_count = count_list[j]
        unique_vals = range_list[j]
        if min_val == 0 and max_val == 1 and unique_count != 2:
            # 0-1连续特征：步长0.1
            x_list.append([i / 10 for i in range(11)])
        elif (max_val - min_val) / unique_count > 100:
            # 稀疏离散特征：补充30个随机值
            补充随机值 = [random.randint(min(unique_vals), max(unique_vals)) for _ in range(30)]
            combined = sorted(list(set(unique_vals + 补充随机值)))
            x_list.append(combined)
        else:
            # 密集离散特征：遍历所有整数
            x_list.append(list(range(int(min_val), int(max_val) + 1)))
    return x_list


# 关键：子进程函数（修复GPU初始化，复用主进程权重）
def process_single_feature_optimized(j, column_names, min_list, max_list, count_list, x_list, base_x, init_val):
    """优化后的子进程函数：不重复初始化GPU，复用全局worker_model"""
    global worker_model
    import tensorflow as tf

    try:
        # 1. 生成当前特征的扰动值（优化：避免重复计算）
        feat_name = column_names[j]
        min_val = min_list[j]
        max_val = max_list[j]
        unique_count = count_list[j]
        if min_val == 0 and max_val == 1 and unique_count != 2:
            values = [i / 10 for i in range(6)]  # 优化：减少计算量（步长0.2）
        elif (max_val - min_val) / unique_count > 100:
            values = random.sample(x_list[j], 15)  # 优化：随机采样15个值
            values.sort()
        elif len(x_list[j]) > 100:
            values = np.linspace(min_val, max_val, 20, dtype=int).tolist()  # 优化：均匀采样20个值
        else:
            values = x_list[j]
        num_values = len(values)

        # 2. 批量生成输入（避免循环）
        modified_x_batch = np.repeat(base_x, repeats=num_values, axis=0)
        modified_x_batch[:, j, 0] = values  # 修改第j个特征
        modified_x_batch_tensor = tf.convert_to_tensor(modified_x_batch, dtype=tf.float32)

        # 3. 批量预测（用predict_on_batch减少开销）
        rel_batch = worker_model.predict_on_batch(modified_x_batch_tensor)

        # 4. 计算预测变化量
        rel_out = np.argmax(init_val, axis=1)[0]
        variety_batch = rel_batch - init_val
        first_value_batch = variety_batch[:, rel_out]

        return j, first_value_batch.tolist()

    except Exception as e:
        # 关键：捕获所有异常，返回非空默认值（避免空序列）
        print(f"子进程{j}处理失败：{str(e)}", flush=True)
        return j, [0.0]  # 返回默认值，避免后续max()报错


# 关键：子进程初始化函数（仅加载模型权重，不重复初始化GPU）
def init_worker(weights, input_shape):
    global worker_model
    # 重构与主进程一致的模型结构
    inp = Input(shape=input_shape)
    x = Conv1D(32, 3, padding="same", activation='tanh')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LeakyReLU(alpha=0.33)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LeakyReLU(alpha=0.33)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LeakyReLU(alpha=0.33)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    g = Flatten()(x)
    output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')(g)
    worker_model = Model(inputs=inp, outputs=output)
    # 加载主进程传递的权重（避免重复读取磁盘）
    worker_model.set_weights(weights)
    # 预热计算图（减少首次预测开销）
    warmup_input = tf.random.normal((8, input_shape[0], input_shape[1]), dtype=tf.float32)
    worker_model.predict_on_batch(warmup_input)
    print("子进程：模型初始化完成", flush=True)


def Fresh_breeze_parallel(x, column_names, min_list, max_list, count_list, init_val, x_list, model_weights,
                          input_shape):
    y_list = [[] for _ in range(len(column_names))]
    start = time.perf_counter()

    # 预处理基础数据（主进程一次处理，子进程复用）
    base_x = x.numpy()  # shape=(1, n_features, 1)
    n_features = len(column_names)

    # 关键：减少进程数（避免GPU内存竞争，根据GPU内存调整，建议≤2）
    num_processes = min(mp.cpu_count() // 4, 2)  # 从4改为2，降低内存占用
    print(f"使用 {num_processes} 个进程进行特征重要性评估（降低GPU负载）", flush=True)

    # 进程池：传递模型权重和输入形状，避免子进程重复加载模型
    with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(model_weights, input_shape)  # 传递权重而非模型路径
    ) as pool:
        # 生成任务列表
        tasks = [
            (j, column_names, min_list, max_list, count_list, x_list, base_x, init_val)
            for j in range(n_features)
        ]
        # 批量执行任务
        results = pool.starmap(process_single_feature_optimized, tasks)

    # 整理结果（按特征索引填充）
    for j, first_value_batch in results:
        y_list[j] = first_value_batch  # 即使失败，也有默认值[0.0]

    end = time.perf_counter()
    print(f'Fresh_breeze多进程耗时: {end - start:.4f} Seconds', flush=True)
    return y_list


def visualization(column_names, y_list, x_list):
    # 1. 预处理：用0.0填充空序列，避免后续计算报错
    y_list = [sublist if sublist else [0.0] for sublist in y_list]
    # 计算每个特征的最大/最小变化值（用于设置y轴范围和重要性计算）
    y_list_max = []
    y_list_min = []
    for sublist in y_list:
        # 处理极端值（避免y轴范围过大）
        sub_max = max(sublist) if sublist else 0.0
        sub_min = min(sublist) if sublist else 0.0
        # 若所有值为0，手动设置微小范围（避免ylim出错）
        if sub_max == sub_min == 0.0:
            sub_max = 0.001
            sub_min = -0.001
        y_list_max.append(sub_max)
        y_list_min.append(sub_min)
    # 计算特征重要性（最大绝对变化值）
    y_list_abs = [[abs(x) for x in inner] for inner in y_list]
    max_values = [max(sublist) for sublist in y_list_abs]

    # 2. 分页配置：每组最多显示50个特征（可根据需求调整）
    page_size = 50  # 每页最大子图数量
    total_features = len(column_names)
    total_pages = (total_features + page_size - 1) // page_size  # 向上取整计算总页数

    # 3. 分页绘制子图
    for page in range(total_pages):
        # 计算当前页的特征范围（避免越界）
        start_idx = page * page_size
        end_idx = min((page + 1) * page_size, total_features)
        # 当前页的特征、y值、x值
        current_cols = column_names[start_idx:end_idx]
        current_y = y_list[start_idx:end_idx]
        current_x = x_list[start_idx:end_idx]
        current_ymax = y_list_max[start_idx:end_idx]
        current_ymin = y_list_min[start_idx:end_idx]
        current_page_num = page + 1  # 页码从1开始

        # 创建画布：宽度4英寸，高度=子图数量×0.8英寸（缩小单图高度）
        fig_height = len(current_cols) * 0.8
        fig, axs = plt.subplots(
            nrows=len(current_cols), ncols=1,
            figsize=(4, fig_height),  # 关键：缩小高度
            squeeze=False  # 强制返回2D数组，兼容单特征情况
        )
        axs = axs.flatten()  # 转为1D数组，方便循环

        # 绘制当前页的每个特征
        for i in range(len(current_cols)):
            ax = axs[i]
            feat_name = current_cols[i]
            x_vals = current_x[i]
            y_vals = current_y[i]

            # 确保x和y长度一致（避免绘图报错）
            min_len = min(len(x_vals), len(y_vals))
            x_vals = x_vals[:min_len]
            y_vals = y_vals[:min_len]

            # 绘制折线图（简化样式，避免冗余）
            ax.plot(x_vals, y_vals, linestyle='-', color='#2E86AB', linewidth=1.2)

            # 标注最大/最小值（仅当有有效数据时）
            if min_len > 0:
                max_idx = np.argmax(y_vals)
                min_idx = np.argmin(y_vals)
                # 最大値（红色）
                ax.annotate(
                    f'Max: {y_vals[max_idx]:.3f}',
                    (x_vals[max_idx], y_vals[max_idx]),
                    textcoords="offset points", xytext=(0, 3), ha='center',
                    color='#A23B72', fontsize=4, weight='bold'
                )
                # 最小値（绿色）
                ax.annotate(
                    f'Min: {y_vals[min_idx]:.3f}',
                    (x_vals[min_idx], y_vals[min_idx]),
                    textcoords="offset points", xytext=(0, -6), ha='center',
                    color='#F18F01', fontsize=4
                )

            # 设置子图样式（缩小字体，避免重叠）
            ax.set_xlabel(feat_name, fontsize=5, labelpad=2)
            ax.set_ylabel('Forecast Change', fontsize=5, labelpad=2)
            ax.set_ylim(current_ymin[i] * 1.1, current_ymax[i] * 1.5)  # 预留少量空间
            ax.grid(True, alpha=0.2, linewidth=0.5)
            # 缩小刻度字体
            ax.tick_params(axis='both', labelsize=4, pad=1)
            # 移除顶部和右侧边框（简洁样式）
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 调整子图间距（垂直间距0.1英寸，避免重叠）
        plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.5)
        # 保存当前页图片（按页码命名，避免覆盖）
        save_path = f'feature_changes_page_{current_page_num}.png'
        plt.savefig(
            save_path,
            dpi=150,  # 保持清晰度，同时控制尺寸
            bbox_inches='tight',  # 裁剪多余空白
            facecolor='white'  # 背景色设为白色（避免透明）
        )
        print(f'已保存第{current_page_num}/{total_pages}页特征变化图：{save_path}')
        plt.close(fig)  # 关闭当前画布，释放内存

    return max_values  # 返回重要性值，不影响后续流程


def feature_important(column_names, threshold, max_values):  # Visual feature importance analysis
    total_sum = sum(max_values)
    if total_sum == 0:
        print("警告：所有特征重要性为0，无法绘制图表", flush=True)
        return
    # 计算百分比并合并小特征
    percentages = [(v / total_sum) * 100 for v in max_values]
    merged_values = []
    merged_categories = []
    other_total = 0
    for cat, val in zip(column_names, max_values):
        if (val / total_sum) * 100 < threshold:
            other_total += val
        else:
            merged_values.append(val)
            merged_categories.append(cat)
    # 添加"其他"类别
    if other_total > 0:
        merged_values.append(other_total)
        merged_categories.append("others")
    # 绘制图表
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # 折线图+柱状图
    axs[0].plot(merged_categories, merged_values, marker='o', color='#813C85', linewidth=2, markersize=6)
    axs[0].bar(merged_categories, merged_values, alpha=0.4, color='#DE3F7C', width=0.6)
    axs[0].set_xlabel('Features', fontsize=10)
    axs[0].set_ylabel('Importance (Max Change)', fontsize=10)
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].grid(True, alpha=0.3)
    # 饼图
    axs[1].pie(merged_values, labels=merged_categories, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axs[1].set_title('Feature Importance Distribution', fontsize=10)
    # 保存图表
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def feature_selection(column_names, feature_dict, rela_class, max_values):
    # 更新特征的变化范围
    for i, feat in enumerate(column_names):
        feature_dict[feat]['change_range'] = max_values[i]
    # 深拷贝避免修改原始数据
    rela_class_copy = copy.deepcopy(rela_class)
    feature_dict_copy = copy.deepcopy(feature_dict)
    F = column_names
    F_copy = copy.deepcopy(F)
    flag = 0

    # 统计高贡献特征数量（change_range ≥ 0.1）
    high_contrib_count = sum(1 for feat in F if feature_dict[feat]['change_range'] >= 0.1)
    high_contrib_ratio = high_contrib_count / len(F) if len(F) > 0 else 0

    if high_contrib_ratio >= 0.15:
        # Case1：高贡献特征较多，移除极端值和冗余特征
        # 1. 移除change_range异常的特征
        for corr_group in rela_class_copy:
            to_remove = [feat for feat in corr_group if
                         feature_dict[feat]['change_range'] >= 0.5 or feature_dict[feat]['change_range'] <= 0.1]
            for feat in to_remove:
                corr_group.remove(feat)
                # 更新剩余特征的NOFR
                for remaining_feat in corr_group:
                    if feature_dict_copy[remaining_feat]['NOFR'] > 0:
                        feature_dict_copy[remaining_feat]['NOFR'] -= 1
        # 2. 移除低贡献特征
        F_copy = [feat for feat in F_copy if feature_dict[feat]['change_range'] > 0.1]
        # 3. 过滤中等贡献但相关性低的特征
        for feat in F:
            if (0.1 < feature_dict[feat]['change_range'] < 0.5 and
                    -0.3 < feature_dict[feat]['target'] < 0.3 and
                    feature_dict_copy[feat]['NOFR'] > 0):
                # 找到特征所在的关联组
                for corr_group in rela_class_copy:
                    if feat in corr_group:
                        flag = 1
                        # 保留关联组中change_range最大的特征
                        group_changes = [feature_dict[f]['change_range'] for f in corr_group]
                        max_change = max(group_changes)
                        max_feat = [f for f in corr_group if feature_dict[f]['change_range'] == max_change][0]
                        # 移除组内其他特征
                        for f in corr_group:
                            feature_dict_copy[f]['NOFR'] = 0
                            if f != max_feat and f in F_copy:
                                F_copy.remove(f)
                        break
                flag = 0
    else:
        # Case2：高贡献特征较少，保留低贡献但相关性高的特征
        # 1. 移除高贡献特征（避免主导选择）
        for corr_group in rela_class_copy:
            to_remove = [feat for feat in corr_group if feature_dict[feat]['change_range'] >= 0.1]
            for feat in to_remove:
                corr_group.remove(feat)
                for remaining_feat in corr_group:
                    if feature_dict_copy[remaining_feat]['NOFR'] > 0:
                        feature_dict_copy[remaining_feat]['NOFR'] -= 1
        # 2. 过滤相关性低的特征
        for feat in F:
            if (-0.7 < feature_dict[feat]['target'] < 0.7 and
                    feature_dict_copy[feat]['NOFR'] > 0):
                for corr_group in rela_class_copy:
                    if feat in corr_group:
                        flag = 1
                        # 保留关联组中change_range最大的特征
                        group_changes = [feature_dict[f]['change_range'] for f in corr_group]
                        max_change = max(group_changes)
                        max_feat = [f for f in corr_group if feature_dict[f]['change_range'] == max_change][0]
                        for f in corr_group:
                            feature_dict_copy[f]['NOFR'] = 0
                            if f != max_feat and f in F_copy:
                                F_copy.remove(f)
                        break
                flag = 0

    # 计算需要移除的特征
    F_remove = [feat for feat in F if feat not in F_copy]
    print(f"特征选择完成：移除 {len(F_remove)} 个特征，保留 {len(F_copy)} 个特征", flush=True)
    return F_remove


if __name__ == '__main__':
    # 1. 初始化GPU和多进程（主进程统一处理）
    init_gpu()  # 关键：主进程提前初始化GPU，子进程不再修改
    init_multiprocessing()

    # 2. 配置路径
    data_path = "dataset/BrainMethod.csv"  # 注意：数据集名称已修改
    model_save_path = "featureX_model/BrainMethod.keras"  # 模型名称已修改
    targetname = "is_brain"

    # 3. 加载原始数据（无论是否训练模型，都需要原始数据做相关性分析）
    df = pd.read_csv(data_path, encoding="gbk")
    print("原始数据前5行：")
    print(df.head())

    # ---------------------- 新增：处理分类特征（字符串转数值） ----------------------
    print("\n=== 开始处理分类特征 ===")
    # 识别分类特征（object类型或字符串类型）
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # 排除目标变量（如果目标变量在分类特征中）
    if targetname in categorical_features:
        categorical_features.remove(targetname)
    print(f"发现{len(categorical_features)}个分类特征：{categorical_features}")

    # 对分类特征进行编码
    df_encoded = df.copy()  # 避免修改原始数据
    label_encoders = {}  # 保存编码器，便于后续解释

    for feat in categorical_features:
        # 检查特征值是否为字符串类型
        if df[feat].dtype == 'object' and all(isinstance(x, str) for x in df[feat].dropna()):
            # 使用标签编码（适用于大多数场景）
            le = LabelEncoder()
            # 处理可能的缺失值（填充为众数）
            df_encoded[feat] = df_encoded[feat].fillna(df_encoded[feat].mode()[0])
            df_encoded[feat] = le.fit_transform(df_encoded[feat])
            label_encoders[feat] = le
            # 打印编码映射关系（前5个值）
            mappings = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"{feat}的编码映射（部分）：{dict(list(mappings.items())[:5])}")

    # 更新特征和目标变量（使用编码后的数据）
    column_names = df_encoded.columns[df_encoded.columns != targetname].tolist()
    x_original = df_encoded[column_names]
    y_original = df_encoded[targetname]
    # ---------------------- 分类特征处理结束 ----------------------

    # ---------------------- 模型文件存在性判断 ----------------------
    model_exists = os.path.exists(model_save_path)
    if model_exists:
        print(f"\n=== 发现已存在模型文件：{model_save_path} ===", flush=True)
        print("=== 跳过模型训练，直接加载模型并进入相关性分析 ===", flush=True)
        # 加载已有的模型
        model = load_model(model_save_path)
        # 获取模型输入形状（后续特征重要性评估需用到）
        input_shape = model.input_shape[1:]  # 跳过batch维度，格式为(特征数, 1)
        # 过采样数据（后续用选择后的特征重新训练时需用到，提前处理）
        from imblearn.over_sampling import SVMSMOTE

        sm = SVMSMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x_original, y_original)
    else:
        print(f"\n=== 未发现模型文件：{model_save_path} ===", flush=True)
        print("=== 执行完整模型训练流程 ===", flush=True)
        # 3. 处理类别不平衡（仅训练时需要）
        from imblearn.over_sampling import SVMSMOTE

        sm = SVMSMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x_original, y_original)

        # 4. 数据预处理（划分训练集/测试集，reshape为1D卷积输入）
        x_train, x_test, y_train, y_test = train_test_split(
            x_resampled, y_resampled, test_size=0.2, random_state=0
        )
        # reshape为 (样本数, 特征数, 1)（1D卷积要求的输入形状）
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))
        # 标签One-Hot编码
        onehot = OneHotEncoder(sparse_output=False)
        y_train = onehot.fit_transform(y_train.values.reshape(-1, 1))
        y_test = onehot.transform(y_test.values.reshape(-1, 1))  # 测试集用训练集的编码器，避免数据泄露

        # 5. 构建1D卷积模型
        input_shape = x_train.shape[1:]  # (特征数, 1)
        inp = Input(shape=input_shape)
        x = Conv1D(32, 3, padding="same", activation='tanh')(inp)
        x = BatchNormalization()(x)
        x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        g = Flatten()(x)
        output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')(g)

        model = Model(inputs=inp, outputs=output)
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 6. 训练模型（添加学习率衰减）
        lr_reduce = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6
        )
        print("开始训练模型...", flush=True)
        start_train = time.perf_counter()
        history = model.fit(
            x_train, y_train,
            epochs=30,
            callbacks=[lr_reduce],
            batch_size=256,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1
        )
        end_train = time.perf_counter()
        print(f"模型训练耗时: {end_train - start_train:.4f} Seconds", flush=True)

        # 7. 模型评估（仅新训练时执行）
        print("\n模型评估结果:", flush=True)
        loss, acc = model.evaluate(x_test, y_test, verbose=1)
        print(f"测试集Loss: {loss:.4f}, Accuracy: {acc:.4f}", flush=True)
        # 分类报告
        y_pred = model.predict(x_test, verbose=0)
        y_pred_idx = np.argmax(y_pred, axis=1)
        y_test_idx = np.argmax(y_test, axis=1)
        from sklearn.metrics import classification_report

        print("分类报告:", flush=True)
        print(classification_report(y_test_idx, y_pred_idx, digits=5), flush=True)

        # 8. 保存模型（确保文件夹存在）
        model_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"创建模型保存文件夹: {model_dir}", flush=True)
        model.save(model_save_path)
        print(f"模型保存成功: {model_save_path}", flush=True)
    # ---------------------- 模型文件判断逻辑结束 ----------------------

    # ---------------------- 核心模块：特征重要性分析 ----------------------
    total_time = 0.0

    # 9. 1. 相关性分析模块
    print("\n=== 开始相关性分析 ===", flush=True)
    corr_start = time.perf_counter()
    # 初始化特征字典
    feature_dict = {}
    feature_dict = set_feature_dict(column_names, feature_dict)
    # 特征间Spearman相关性分析（用原始数据，避免过采样影响相关性计算）
    spearmanr_data = df_encoded[column_names]  # 使用编码后的数据
    rela_class, feature_dict, corr_list = Feature_correlation_analysis(column_names, feature_dict, spearmanr_data)
    # 特征与目标变量相关性分析（用原始目标变量）
    target_variable = df_encoded[targetname]

    # 目标变量字符串转数值（如果需要）
    print(f"原始目标变量类型: {target_variable.dtype}, 前5个值: {target_variable.head().tolist()}", flush=True)
    if pd.api.types.is_object_dtype(target_variable) or pd.api.types.is_string_dtype(target_variable):
        codes, unique_labels = pd.factorize(target_variable)
        target_variable = pd.Series(codes, index=target_variable.index)
        print(f"目标变量字符串映射关系: {dict(zip(unique_labels, range(len(unique_labels))))}", flush=True)
    target_variable = pd.to_numeric(target_variable, errors='coerce')
    if isinstance(target_variable, pd.Series):
        print(f"转换后目标变量类型: {target_variable.dtype}, 前5个值: {target_variable.head().tolist()}", flush=True)
    else:
        print(f"转换后目标变量类型: {target_variable.dtype}, 前5个值: {target_variable[:5].tolist()}", flush=True)

    # 原代码：feature_dict, target_corr_list = target_correlation_analysis(feature_dict, target_variable, column_names)
    # 修改后（新增df_encoded参数）：
    feature_dict, target_corr_list = target_correlation_analysis(feature_dict, target_variable, column_names,
                                                                 df_encoded)
    # 计时
    corr_end = time.perf_counter()
    corr_time = corr_end - corr_start
    total_time += corr_time
    print(f"相关性分析耗时: {corr_time:.4f} Seconds", flush=True)

    # 10. 2. 特征扰动模块（生成特征的扰动值范围）
    print("\n=== 开始特征扰动 ===", flush=True)
    disturb_start = time.perf_counter()
    # 获取特征统计信息（用编码后的数据计算）
    min_list, max_list, count_list, range_list = value_range(column_names, df_encoded)
    # 生成扰动值范围
    x_list = Disturbance_range(column_names, min_list, max_list, count_list, range_list)
    # 计时
    disturb_end = time.perf_counter()
    disturb_time = disturb_end - disturb_start
    total_time += disturb_time
    print(f"特征扰动耗时: {disturb_time:.4f} Seconds", flush=True)

    # 11. 3. 重要性评估模块（多进程加速）
    print("\n=== 开始重要性评估 ===", flush=True)
    importance_start = time.perf_counter()
    # 准备基础输入（用编码后数据的特征均值，作为扰动基准）
    base_data = df_encoded[column_names].mean().round().astype(int)
    base_x = base_data.values.reshape(1, len(column_names), 1)  # (1, 特征数, 1)
    base_x_tensor = tf.convert_to_tensor(base_x, dtype=tf.float32)
    # 基础输入的初始预测值（用加载/训练后的模型计算）
    init_val = model.predict_on_batch(base_x_tensor)
    # 提取模型权重（传递给子进程）
    model_weights = model.get_weights()
    # 多进程计算特征重要性
    y_list = Fresh_breeze_parallel(
        base_x_tensor,  # 基础输入张量
        column_names,  # 特征名称
        min_list,  # 特征最小值列表
        max_list,  # 特征最大值列表
        count_list,  # 特征唯一值数量列表
        init_val,  # 初始预测值
        x_list,  # 扰动值范围列表
        model_weights,  # 模型权重（子进程复用）
        input_shape  # 模型输入形状
    )
    # 可视化特征变化并计算最大重要性
    max_values = visualization(column_names, y_list, x_list)
    # 绘制特征重要性图表（阈值2%）
    feature_important(column_names, threshold=2, max_values=max_values)
    # 计时
    importance_end = time.perf_counter()
    importance_time = importance_end - importance_start
    total_time += importance_time
    print(f"重要性评估耗时: {importance_time:.4f} Seconds", flush=True)

    # 12. 4. 特征选择模块（移除冗余/低重要性特征）
    print("\n=== 开始特征选择 ===", flush=True)
    selection_start = time.perf_counter()
    # 选择需要移除的特征
    F_remove = feature_selection(column_names, feature_dict, rela_class, max_values)
    # 计时
    selection_end = time.perf_counter()
    selection_time = selection_end - selection_start
    total_time += selection_time
    print(f"特征选择耗时: {selection_time:.4f} Seconds", flush=True)

    # 13. 用选择后的特征重新训练模型（验证选择效果，无论原模型是否加载都执行）
    print("\n=== 用选择后的特征重新训练模型 ===", flush=True)
    # 移除低重要性特征（用之前过采样的数据）
    x_selected = x_resampled.drop(columns=F_remove) if len(F_remove) > 0 else x_resampled
    # 划分训练集/测试集
    x_train_sel, x_test_sel, y_train_sel, y_test_sel = train_test_split(
        x_selected, y_resampled, test_size=0.2, random_state=0
    )
    # reshape为1D卷积输入
    x_train_sel = x_train_sel.values.reshape((x_train_sel.shape[0], x_train_sel.shape[1], 1))
    x_test_sel = x_test_sel.values.reshape((x_test_sel.shape[0], x_test_sel.shape[1], 1))
    # 标签One-Hot编码（若之前训练过模型，复用onehot；否则重新初始化）
    try:
        # 尝试复用已有的onehot编码器（避免重新拟合）
        y_train_sel = onehot.transform(y_train_sel.values.reshape(-1, 1))
        y_test_sel = onehot.transform(y_test_sel.values.reshape(-1, 1))
    except NameError:
        # 若onehot未定义（即加载模型的情况），重新初始化编码器
        onehot_new = OneHotEncoder(sparse_output=False)
        y_train_sel = onehot_new.fit_transform(y_train_sel.values.reshape(-1, 1))
        y_test_sel = onehot_new.transform(y_test_sel.values.reshape(-1, 1))

    # 构建新模型（输入形状适配选择后的特征数）
    new_input_shape = x_train_sel.shape[1:]
    new_inp = Input(shape=new_input_shape)
    new_x = Conv1D(32, 3, padding="same", activation='tanh')(new_inp)
    new_x = BatchNormalization()(new_x)
    new_x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(new_x)
    new_x = LeakyReLU(alpha=0.33)(new_x)
    new_x = Dropout(0.5)(new_x)
    new_x = BatchNormalization()(new_x)
    new_x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(new_x)
    new_x = LeakyReLU(alpha=0.33)(new_x)
    new_x = Dropout(0.5)(new_x)
    new_x = BatchNormalization()(new_x)
    new_x = Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(new_x)
    new_x = LeakyReLU(alpha=0.33)(new_x)
    new_x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(new_x)
    new_g = Flatten()(new_x)
    new_output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')(new_g)

    new_model = Model(inputs=new_inp, outputs=new_output)
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练新模型
    lr_reduce_new = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6
    )
    start_new_train = time.perf_counter()
    new_history = new_model.fit(
        x_train_sel, y_train_sel,
        epochs=30,
        callbacks=[lr_reduce_new],
        batch_size=256,
        validation_data=(x_test_sel, y_test_sel),
        shuffle=True,
        verbose=1
    )
    end_new_train = time.perf_counter()
    print(f"新模型训练耗时: {end_new_train - start_new_train:.4f} Seconds", flush=True)

    # 评估新模型
    print("\n新模型评估结果:", flush=True)
    new_loss, new_acc = new_model.evaluate(x_test_sel, y_test_sel, verbose=1)
    print(f"新模型测试集Loss: {new_loss:.4f}, Accuracy: {new_acc:.4f}", flush=True)
    # 新模型分类报告
    new_y_pred = new_model.predict(x_test_sel, verbose=0)
    new_y_pred_idx = np.argmax(new_y_pred, axis=1)
    new_y_test_idx = np.argmax(y_test_sel, axis=1)
    print("新模型分类报告:", flush=True)
    print(classification_report(new_y_test_idx, new_y_pred_idx, digits=5), flush=True)

    # 14. 总计时间汇总
    print("\n" + "=" * 60)
    print("特征重要性分析四大模块耗时汇总:")
    print(f"1. 相关性分析: {corr_time:.4f} Seconds")
    print(f"2. 特征扰动: {disturb_time:.4f} Seconds")
    print(f"3. 重要性评估: {importance_time:.4f} Seconds")
    print(f"4. 特征选择: {selection_time:.4f} Seconds")
    print(f"{'=' * 60}")
    print(f"四大模块总计耗时: {total_time:.4f} Seconds")
    print("=" * 60)
