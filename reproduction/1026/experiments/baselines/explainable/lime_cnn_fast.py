# 新增：放在所有 import 最前面，验证版本
import imblearn

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SVMSMOTE
from multiprocessing import Pool, cpu_count
# -------------------------- 1. 替换SHAP为LIME依赖 --------------------------
from lime import lime_tabular  # LIME表格数据解释器
import matplotlib.pyplot as plt  # LIME可视化依赖
from concurrent.futures import ThreadPoolExecutor  # 并行计算LIME解释

# -------------------------- 基础配置与数据读取（混合精度保留） --------------------------
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from tqdm import tqdm


# 关键修复1：仅在LIME阶段启用Eager，训练阶段禁用（避免与MirroredStrategy冲突）
# 先默认禁用Eager，后续LIME计算时再临时启用
tf.config.run_functions_eagerly(False)
print(f"✅ 初始禁用Eager Execution，避免多GPU策略冲突")

# 1. 先初始化混合精度策略（确保加载模型时也适配）
policy = Policy('mixed_float16')
set_global_policy(policy)
print(f"✅ 混合精度策略已启用：{policy.name}")
print(f"✅ 计算 dtype: {tf.keras.backend.floatx()}")
print(f"✅ 变量 dtype: {policy.variable_dtype}")

# 2. 路径配置（模型保存/加载核心路径）
datapath = "dataset/SoftwareQuality.csv"
dataset_filename = datapath.split("/")[-1]
dataset_name = os.path.splitext(dataset_filename)[0]
model_save_dir = "lime_model"  # 模型保存目录
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f"{dataset_name}_model.keras")  # 模型保存路径
# 筛选后模型单独保存路径（避免覆盖原模型）
model_f_save_path = os.path.join(model_save_dir, f"{dataset_name}_filtered_model.keras")
print(f"📂 原模型路径：{model_save_path}")
print(f"📂 筛选后模型路径：{model_f_save_path}")

# -------------------------- 关键修复1：数据清洗（处理NaN/异常值） --------------------------
# 数据预处理：无论模型是否加载，都需准备数据（评估/LIME需用）
df = pd.read_csv(datapath, encoding="gbk")
print(f"🔍 原始数据形状：{df.shape}")
print(f"🔍 NaN值统计：\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df = df.fillna(df.mean(numeric_only=True))
assert df.isnull().sum().sum() == 0, "❌ 数据仍存在NaN值，请检查填充逻辑！"

x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
column_names = df.columns[1:-1].tolist()
num_features = len(column_names)  # 固定特征数，后续校验用
num_classes = len(y.unique())
class_names = sorted(y.unique().tolist())
print(f"📊 数据基本信息：特征数{num_features}，类别数{num_classes}，类别标签{class_names}")

# -------------------------- 并行过采样处理（新增缓存，避免重复计算） --------------------------
cache_save_dir = "npz"
cache_path = os.path.join(cache_save_dir, f"{dataset_name}_preprocessed_data.npz")

# 创建缓存目录（如果不存在）
os.makedirs(cache_save_dir, exist_ok=True)

if os.path.exists(cache_path):
    print("🔍 发现缓存，直接加载过采样数据...")
    data = np.load(cache_path)
    # 关键修改：将数组转为 DataFrame（使用原始列名）
    x_resampled = pd.DataFrame(data['x_resampled'], columns=column_names)
    y_resampled = data['y_resampled']  # y 保持数组格式
else:
    print("🚀 无缓存，执行SVMSMOTE过采样（仅1次）...")
    sm = SVMSMOTE(random_state=42)
    for _ in tqdm([0], desc="SVMSMOTE过采样"):
        x_resampled, y_resampled = sm.fit_resample(x, y)
    # 新生成的数据中，x 是 DataFrame，y 转为数组
    y_resampled = y_resampled.to_numpy()
    np.savez(cache_path, x_resampled=x_resampled, y_resampled=y_resampled)
    print(f"💾 过采样数据已缓存至：{cache_path}")

# 数据拆分
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.2, random_state=0
)

# 维度调整（模型输入3D + LIME输入2D）
x_train_3d = x_train.to_numpy().reshape((-1, num_features, 1))  # 用固定特征数
x_test_3d = x_test.to_numpy().reshape((-1, num_features, 1))
x_train_2d = x_train.to_numpy()
x_test_2d = x_test.to_numpy()

# LIME输入校验（强化校验）
assert len(x_train_2d.shape) == 2, f"❌ x_train_2d需为2D数组，当前{len(x_train_2d.shape)}D"
assert x_train_2d.shape[1] == num_features, f"❌ x_train_2d特征数{x_train_2d.shape[1]}≠{num_features}"
assert len(x_test_2d.shape) == 2, f"❌ x_test_2d需为2D数组，当前{len(x_test_2d.shape)}D"
assert x_test_2d.shape[1] == num_features, f"❌ x_test_2d特征数{x_test_2d.shape[1]}≠{num_features}"
print(f"✅ LIME输入校验通过：x_train_2d={x_train_2d.shape}, x_test_2d={x_test_2d.shape}")

# 标签编码（修复参数名，消除FutureWarning）
onehot = OneHotEncoder(sparse_output=False)
y_train_onehot = onehot.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = onehot.transform(y_test.reshape(-1, 1))
print(f"📥 训练集形状：{x_train_3d.shape}（模型输入）/{x_train_2d.shape}（LIME输入）")

# -------------------------- 修复：日志过滤（避免空列表索引错误） --------------------------
# 先获取当前日志器的过滤器列表
logger = tf.get_logger()
current_filters = logger.filters

# 方案1：若已有过滤器，基于第一个过滤器类继承；若无，直接继承基础Filter类
if current_filters:
    base_filter_class = current_filters[0].__class__
else:
    from logging import Filter  # 导入Python原生日志Filter基类
    base_filter_class = Filter

# 定义自定义日志过滤器（屏蔽指定关键词日志）
class DataLogFilter(base_filter_class):
    def filter(self, record):
        # 屏蔽包含以下关键词的日志（避免冗余输出）
        unwanted_keywords = ["experimental_type", "auto_shard.cc"]
        if any(keyword in record.getMessage() for keyword in unwanted_keywords):
            return False  # 不输出该日志
        # 若继承自已有过滤器，需保留其原有过滤逻辑
        return super().filter(record) if current_filters else True

# 清空原有过滤器（避免重复过滤），添加自定义过滤器
logger.handlers[0].filters = [DataLogFilter()]
print("✅ 日志过滤器初始化完成（已屏蔽指定关键词日志）")
# -------------------------- 数据加载优化（tf.data保留） --------------------------
def create_tf_dataset(x, y, batch_size, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_train:
        # 多线程并行shuffle（平衡内存与速度）
        dataset = dataset.shuffle(buffer_size=min(len(x)//10, 10000), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    # 多线程并行批次处理
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    # 预加载4个批次，避免GPU等待
    dataset = dataset.prefetch(buffer_size=batch_size * 4)
    return dataset


# 多GPU配置（无论加载/训练，都需配置批次大小）
base_batch_size = 128
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync
batch_size = base_batch_size * num_gpus
print(f"🖥️ 使用{num_gpus}个GPU，最终批次大小：{batch_size}")

# 创建tf.data数据集（评估/训练均需）
train_dataset = create_tf_dataset(x_train_3d, y_train_onehot, batch_size, is_train=True)
val_dataset = create_tf_dataset(x_test_3d, y_test_onehot, batch_size, is_train=False)

# 计算训练/评估步数（加载模型后评估需用）
steps_per_epoch = len(x_train_3d) // batch_size
val_steps = len(x_test_3d) // batch_size
print(f"🚩 计算步数：steps_per_epoch={steps_per_epoch}, val_steps={val_steps}")

# -------------------------- 核心修改：模型加载/训练分支 --------------------------
# 检查模型是否已存在
if os.path.exists(model_save_path):
    print(f"🔍 发现已保存模型，正在加载...")
    # 加载模型（自动保留训练时的混合精度、编译配置）
    model = tf.keras.models.load_model(model_save_path)
    print(f"✅ 模型加载成功！路径：{model_save_path}")
else:
    print(f"⚠️  未发现已保存模型，开始首次训练...")
    # 多GPU并行模型训练（仅首次执行）
    with strategy.scope():
        inp = Input(shape=(num_features, 1))  # 用固定特征数，避免维度歧义
        x = Conv1D(32, 6, padding="same")(inp)
        x = BatchNormalization()(x)
        x = tf.keras.activations.tanh(x)
        x = Dropout(0.5)(x)

        x = Conv1D(64, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Conv1D(64, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(0.5)(x)

        x = Conv1D(128, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Conv1D(128, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(0.5)(x)

        x = Conv1D(128, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Conv1D(128, 6, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(0.5)(x)

        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Flatten()(x)
        output = Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='softmax',
            dtype='float32'
        )(x)
        model = Model(inputs=inp, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # 训练原模型的回调（简化，避免参数冲突）
    orig_callbacks = [
        keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1, monitor='val_loss'),
    ]

    # 首次训练
    print(f"🚀 开始首次训练（epochs=30，早停兜底）")
    start = time.perf_counter()
    history = model.fit(
        train_dataset,
        epochs=30,
        callbacks=orig_callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps,
        shuffle=False,
        verbose=1
    )
    end = time.perf_counter()

    # 首次训练后保存模型
    model.save(model_save_path)
    print(f"💾 首次训练完成，模型已保存至：{model_save_path}")
    print(f"⏱️  首次训练总耗时：{end - start:.2f} 秒")

# -------------------------- 模型评估（加载/训练后统一执行） --------------------------
print(f"\n📊 开始模型评估...")
val_loss, val_acc = model.evaluate(val_dataset, steps=val_steps)
print(f"📈 验证集性能：损失{val_loss:.4f}，精度{val_acc:.4f}")

# 分类报告（加载/训练后统一执行）
Y_test = np.argmax(y_test_onehot, axis=1)
predict = model.predict(val_dataset, steps=val_steps)
y_pred = np.argmax(predict, axis=1)
print("\n📋 分类报告：")
print(classification_report(Y_test[:len(y_pred)], y_pred, digits=5))

# -------------------------- LIME特征重要性分析（临时启用Eager） --------------------------
print("\n🔥 开始计算LIME特征重要性（CPU并行+缓存）...")
lime_cache_path = f"dataset/lime_importance_{dataset_name}.npz"
lime_vis_path = f"dataset/lime_importance_plot_{dataset_name}.png"

# 关键修复2：LIME计算时临时启用Eager（仅LIME阶段用）
tf.config.run_functions_eagerly(True)
print(f"✅ LIME计算阶段临时启用Eager Execution")

# 从LIME离散化特征名中提取原始特征名
def extract_original_feat_name(lime_feat_name, original_names):
    for orig_name in original_names:
        if orig_name in lime_feat_name:
            return orig_name
    raise ValueError(f"❌ 从LIME特征名「{lime_feat_name}」中未找到原始特征名（原始列表：{original_names}）")

# 严格控制输入维度，过滤无效样本
def lime_predict_fn(x_2d):
    try:
        # 基础维度校验与修复
        if len(x_2d.shape) != 2:
            raise ValueError(f"输入需为2D数组，当前{len(x_2d.shape)}D")
        if x_2d.shape[1] != num_features:
            if x_2d.shape[1] > num_features:
                x_2d = x_2d[:, :num_features]
            else:
                pad_width = ((0, 0), (0, num_features - x_2d.shape[1]))
                x_2d = np.pad(x_2d, pad_width, mode='constant', constant_values=0)
            print(f"⚠️  修正样本特征数：{x_2d.shape[1]}→{num_features}")

        # 过滤空样本
        valid_mask = x_2d.shape[0] > 0
        if not valid_mask:
            raise ValueError(f"输入样本数为0，无效")
        x_2d_valid = x_2d[valid_mask] if len(x_2d.shape) > 1 else x_2d.reshape(1, -1)

        # 转换为3D输入并预测
        x_3d = x_2d_valid.reshape((-1, num_features, 1))
        preds = model(x_3d, training=False).numpy()

        # 校验输出维度
        assert len(preds.shape) == 2, f"预测输出需为2D数组，当前{len(preds.shape)}D"
        assert preds.shape[0] == x_3d.shape[0], f"预测样本数{preds.shape[0]}≠输入{x_3d.shape[0]}"
        assert preds.shape[1] == num_classes, f"预测类别数{preds.shape[1]}≠{num_classes}"

        return preds

    except Exception as e:
        print(f"⚠️  预测函数临时错误：{str(e)}，返回默认概率")
        default_preds = np.ones((max(1, x_2d.shape[0]), num_classes)) / num_classes
        return default_preds

# LIME计算主逻辑
if os.path.exists(lime_cache_path):
    print(f"🔍 发现LIME缓存，加载特征重要性...")
    cache_data = np.load(lime_cache_path)
    lime_feature_importance = cache_data["importance"]
    if len(lime_feature_importance) != num_features:
        print(f"⚠️ LIME缓存特征数不匹配，重新计算...")
        os.remove(lime_cache_path)
    else:
        start_lime = end_lime = time.perf_counter()
else:
    print("🚀 无LIME缓存，开始计算局部解释...")
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=x_train_2d,
        feature_names=column_names,
        class_names=class_names,
        mode="classification",
        random_state=42,
        discretize_continuous=True,
        discretizer="quartile",
        sample_around_instance=True
    )

    sample_size = min(200, len(x_test_2d))
    sample_indices = np.random.choice(len(x_test_2d), size=sample_size, replace=False)
    x_test_sample = x_test_2d[sample_indices]
    print(f"✅ LIME采样样本形状：{x_test_sample.shape}（样本数：{sample_size}）")

    def explain_single_sample(idx):
        try:
            sample = x_test_sample[idx]
            sample_2d = sample.reshape(1, -1)
            exp = explainer.explain_instance(
                data_row=sample_2d[0],
                predict_fn=lime_predict_fn,
                num_features=num_features,
                top_labels=1,
                num_samples=500
            )

            feature_contrib = np.zeros(num_features)
            for lime_feat_name, contrib in exp.as_list(label=exp.top_labels[0]):
                orig_feat_name = extract_original_feat_name(lime_feat_name, column_names)
                feat_idx = column_names.index(orig_feat_name)
                feature_contrib[feat_idx] = abs(contrib)

            if np.sum(feature_contrib) < 1e-6:
                print(f"⚠️  样本{idx}贡献值过小，视为无效")
                return np.zeros(num_features)
            return feature_contrib

        except Exception as e:
            print(f"⚠️  样本{idx}解释临时错误：{str(e)}，返回零贡献")
            return np.zeros(num_features)

    # 降低线程数，避免GPU竞争
    start_lime = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        all_contrib = list(tqdm(
            executor.map(explain_single_sample, range(sample_size)),
            total=sample_size,
            desc="LIME样本解释"
        ))
    end_lime = time.perf_counter()

    # 过滤无效样本并保存缓存
    valid_contrib = [c for c in all_contrib if np.sum(c) > 1e-6]
    if len(valid_contrib) == 0:
        raise ValueError("❌ 所有样本解释无效，请检查LIME配置或模型输入")
    lime_feature_importance = np.mean(valid_contrib, axis=0)
    np.savez(lime_cache_path, importance=lime_feature_importance)
    print(f"💾 LIME特征重要性已缓存至：{lime_cache_path}（有效样本数：{len(valid_contrib)}/{sample_size}）")

# 关键修复3：LIME计算完成后，禁用Eager以支持多GPU训练
tf.config.run_functions_eagerly(False)
print(f"✅ LIME计算完成，禁用Eager Execution以支持多GPU训练")

print(f"⏱️  LIME计算耗时：{end_lime - start_lime:.2f} 秒")

# -------------------------- LIME可视化（流程不变） --------------------------
print(f"\n📊 生成LIME特征重要性条形图...")
sorted_idx = np.argsort(lime_feature_importance)[::-1]
sorted_feat = [column_names[i] for i in sorted_idx]
sorted_importance = lime_feature_importance[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_feat)), sorted_importance, align="center", color="#1f77b4")
plt.yticks(range(len(sorted_feat)), sorted_feat, fontsize=10)
plt.xlabel("LIME平均绝对贡献值（全局重要性）", fontsize=12)
plt.title(f"LIME Feature Importance (Dataset: {dataset_name})", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(lime_vis_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"💾 LIME可视化图已保存至：{lime_vis_path}")

# -------------------------- 特征筛选与筛选后模型训练（核心修复） --------------------------
total_importance = sum(lime_feature_importance)
if total_importance < 1e-6:
    raise ValueError("❌ LIME特征重要性计算无效，无法进行特征筛选")

lime_importance_ratio = [(imp / total_importance) * 100 for imp in lime_feature_importance]
low_importance_feats = [
    name for i, name in enumerate(column_names)
    if lime_importance_ratio[i] < 0.2
]

print(f"\n🎯 筛选结果：")
print(f"   - 总特征数：{num_features}")
print(f"   - 低重要性特征数：{len(low_importance_feats)}")
print(f"   - 低重要性特征列表：{low_importance_feats[:10]}..." if len(low_importance_feats) > 10
      else f"   - 低重要性特征列表：{low_importance_feats}")

print("\n🚀 开始训练筛选后模型...")
x_filtered = x_resampled.drop(columns=low_importance_feats) if len(low_importance_feats) > 0 else x_resampled.copy()
num_filtered_features = x_filtered.shape[1]

if num_filtered_features == 0:
    raise ValueError("❌ 特征筛选后剩余特征数为0，请降低筛选阈值！")
elif num_filtered_features < 5:
    print(f"⚠️  警告：筛选后仅剩余{num_filtered_features}个特征，可能影响模型精度！")

x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(
    x_filtered, y_resampled, test_size=0.2, random_state=0
)
x_train_f_3d = x_train_f.to_numpy().reshape((-1, num_filtered_features, 1))
x_test_f_3d = x_test_f.to_numpy().reshape((-1, num_filtered_features, 1))
y_train_f_onehot = onehot.transform(y_train_f.reshape(-1, 1))
y_test_f_onehot = onehot.transform(y_test_f.reshape(-1, 1))
print(f"📥 筛选后训练集形状：{x_train_f_3d.shape}（特征数减少{num_features - num_filtered_features}个）")

# 筛选后模型的批次大小调整（动态适配）
batch_size_f = min(base_batch_size * num_gpus, len(x_train_f_3d) // 4)
batch_size_f = max(batch_size_f, 32)
if batch_size_f % 8 != 0:
    batch_size_f = batch_size_f - (batch_size_f % 8)
print(f"🔧 筛选后批次大小调整为：{batch_size_f}")

# 创建筛选后数据集
train_dataset_f = create_tf_dataset(x_train_f_3d, y_train_f_onehot, batch_size_f, is_train=True)
val_dataset_f = create_tf_dataset(x_test_f_3d, y_test_f_onehot, batch_size_f, is_train=False)
steps_per_epoch_f = len(x_train_f_3d) // batch_size_f
val_steps_f = len(x_test_f_3d) // batch_size_f
print(f"🔧 筛选后训练步数：{steps_per_epoch_f}，验证步数：{val_steps_f}")

# 关键修复4：简化回调函数（移除ModelCheckpoint，避免保存参数冲突）
filtered_callbacks = [
    # 每轮验证，确保val_loss可用
    keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1, monitor='val_loss'),
]

# 构建并训练筛选后模型（简化结构，降低计算量）
with strategy.scope():
    inp_f = Input(shape=(num_filtered_features, 1))
    # 减少Conv1D通道数，降低计算量
    x_f = Conv1D(16, 6, padding="same")(inp_f)  # 原32→16
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.tanh(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Conv1D(32, 6, padding="same")(x_f)  # 原64→32
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = Conv1D(32, 6, padding="same")(x_f)  # 原64→32
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = LeakyReLU(alpha=0.33)(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Conv1D(64, 6, padding="same")(x_f)  # 原128→64，移除1个Conv1D块
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = Conv1D(64, 6, padding="same")(x_f)  # 原128→64
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = LeakyReLU(alpha=0.33)(x_f)
    x_f = Dropout(0.5)(x_f)

    # 缩小Dense层维度
    x_f = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_f)  # 原512→256
    x_f = Flatten()(x_f)
    output_f = Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        activation='softmax',
        dtype='float32'
    )(x_f)
    model_f = Model(inputs=inp_f, outputs=output_f)

    model_f.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# 关键修复5：设置validation_freq=1，确保每轮生成val_loss
start_f = time.perf_counter()
history_f = model_f.fit(
    train_dataset_f,
    epochs=30,
    callbacks=filtered_callbacks,
    steps_per_epoch=steps_per_epoch_f,
    validation_data=val_dataset_f,
    validation_steps=val_steps_f,
    shuffle=False,
    verbose=1,
    validation_freq=1  # 每轮验证，确保val_loss可用
)
end_f = time.perf_counter()

# 训练完成后手动保存模型（避免回调冲突）
model_f.save(model_f_save_path)
print(f"💾 筛选后模型已保存至：{model_f_save_path}")

# 筛选后模型评估
val_loss_f, val_acc_f = model_f.evaluate(val_dataset_f, steps=val_steps_f)
print(f"\n📈 筛选后模型验证性能：")
print(f"   - 损失：{val_loss_f:.4f}（原模型：{val_loss:.4f}）")
print(f"   - 精度：{val_acc_f:.4f}（原模型：{val_acc:.4f}）")
print(f"⏱️  筛选后模型训练耗时：{end_f - start_f:.2f} 秒")

# 筛选后模型分类报告
Y_test_f = np.argmax(y_test_f_onehot, axis=1)
predict_f = model_f.predict(val_dataset_f, steps=val_steps_f)
y_pred_f = np.argmax(predict_f, axis=1)
print("\n📋 筛选后模型分类报告：")
print(classification_report(Y_test_f[:len(y_pred_f)], y_pred_f, digits=5))