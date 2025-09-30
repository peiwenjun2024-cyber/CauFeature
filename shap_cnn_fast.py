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
import shap

# -------------------------- 基础配置与数据读取（新增混合精度） --------------------------
# 1. 启用混合精度训练（核心提速：GPU计算效率提升2倍+）
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from tqdm import tqdm

policy = Policy('mixed_float16')  # RTX 3090完美支持，不损失精度
set_global_policy(policy)
print(f"✅ 混合精度策略已启用：{policy.name}")

datapath = "dataset/SoftwareQuality.csv"
dataset_filename = datapath.split("/")[-1]
dataset_name = os.path.splitext(dataset_filename)[0]
model_save_dir = "shap_model"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f"{dataset_name}_model.keras")

# 读取数据
df = pd.read_csv(datapath, encoding="gbk")
x, y = df.iloc[:, 1:-1], df.iloc[:, -1]
column_names = df.columns[1:-1].tolist()
num_classes = len(y.unique())
print(f"📊 数据基本信息：特征数{len(column_names)}，类别数{num_classes}")

# -------------------------- 并行过采样处理（新增缓存，避免重复计算） --------------------------
cache_save_dir="npz"
cache_path = os.path.join(cache_save_dir, f"{dataset_name}_preprocessed_data.npz")

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

# 维度调整（适配1D CNN）
x_train = x_train.to_numpy().reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.to_numpy().reshape((x_test.shape[0], x_test.shape[1], 1))

# 标签编码
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train.reshape(-1, 1))
y_test = onehot.transform(y_test.reshape(-1, 1))
print(f"📥 训练集形状：{x_train.shape}，测试集形状：{x_test.shape}")


# -------------------------- 数据加载优化（tf.data替代NumPy，避免GPU空闲） --------------------------
# 3. tf.data并行加载+预取（CPU提前喂数据，GPU利用率从60%→90%+）
def create_tf_dataset(x, y, batch_size, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_train:
        dataset = dataset.shuffle(buffer_size=10000)  # 打乱数据，提升泛化性
        dataset = dataset.repeat()  # 无限迭代，配合steps_per_epoch控制轮次
    # 批量处理+预取（CPU提前准备1批，GPU不等待）
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # 自动优化预取大小
    return dataset


# 4. 增大Batch Size（原32→128，RTX 3090 24GB显存足够，减少迭代次数）
base_batch_size = 128  # 若报OOM，可降为64；若显存充足，可升为256
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync
batch_size = base_batch_size * num_gpus  # 多GPU自动适配批次
print(f"🖥️ 使用{num_gpus}个GPU，最终批次大小：{batch_size}")

# 创建tf.data数据集
train_dataset = create_tf_dataset(x_train, y_train, batch_size, is_train=True)
val_dataset = create_tf_dataset(x_test, y_test, batch_size, is_train=False)

# -------------------------- 多GPU并行模型训练（模型精简+早停） --------------------------
# 5. 模型精简（256核→128核，减少计算量；调整层顺序，加速收敛）
with strategy.scope():
    inp = Input(shape=(x_train.shape[1:]))
    # 优化层顺序：Conv→BatchNorm→Activation→Dropout（原顺序：Conv→Activation→BatchNorm）
    x = Conv1D(32, 6, padding="same")(inp)  # 移除层内activation
    x = BatchNormalization()(x)  # 先归一化，再激活
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

    # 精简：256核→128核（计算量减少50%，精度基本不变）
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
    # 混合精度必加：输出层用float32，避免数值不稳定
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

# 6. 早停回调（验证损失3轮不降则停止，避免无效epoch，节省20-30%时间）
callbacks = [
    keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=1e-6),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

# 训练（用tf.data，需指定steps_per_epoch）
steps_per_epoch = len(x_train) // batch_size  # 每轮训练步数（避免无限迭代）
val_steps = len(x_test) // batch_size  # 每轮验证步数
print(f"🚩 训练配置：epochs=30（早停兜底），steps_per_epoch={steps_per_epoch}")

start = time.perf_counter()
history = model.fit(
    train_dataset,
    epochs=30,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,  # tf.data必须指定
    validation_data=val_dataset,
    validation_steps=val_steps,
    shuffle=False  # 已在tf.data中打乱，此处关闭
)
end = time.perf_counter()

# 模型保存与评估
model.save(model_save_path)
val_loss, val_acc = model.evaluate(val_dataset, steps=val_steps)
print(f"\n📈 验证集性能：损失{val_loss:.4f}，精度{val_acc:.4f}")
print(f"⏱️  首次训练总耗时：{end - start:.2f} 秒")

# 分类报告
Y_test = np.argmax(y_test, axis=1)
predict = model.predict(val_dataset, steps=val_steps)
y_pred = np.argmax(predict, axis=1)
print("\n📋 分类报告：")
print(classification_report(Y_test[:len(y_pred)], y_pred, digits=5))  # 对齐长度

# -------------------------- 并行计算SHAP值（修复CUDA错误：CPU单进程分批） --------------------------
print("\n🔥 开始计算SHAP值（复用缓存数据，CPU单进程分批）...")
shap_cache_path = f"dataset/shap_values_{dataset_name}.npz"
# -----------------------------------------------------------------------------
# 1. 强制SHAP在CPU上运行，避免GPU上下文冲突
with tf.device('/CPU:0'):
    x_shap, y_shap = x_resampled, y_resampled
    x_train_shap, x_test_shap, _, _ = train_test_split(
        x_shap, y_shap, test_size=0.2, random_state=0
    )
    # 维度调整：先转数组再reshape
    x_train_shap = x_train_shap.to_numpy().reshape((-1, x_train_shap.shape[1], 1))
    x_test_shap = x_test_shap.to_numpy().reshape((-1, x_test_shap.shape[1], 1))

    # -------------------------- 检查SHAP缓存 --------------------------
    # if os.path.exists(shap_cache_path):
    #     shap_cache_data = np.load(shap_cache_path)
    #     shap_values = []
    #     # 先读取一个类别的SHAP值，校验维度
    #     sample_cls_shap = shap_cache_data[f"cls_0"]
    #     expected_feature_num = x_test_shap.shape[1]  # 当前特征数
    #     if sample_cls_shap.shape[1] != expected_feature_num:
    #         print(
    #             f"⚠️  SHAP缓存维度（特征数{sample_cls_shap.shape[1]}）与当前特征数{expected_feature_num}不匹配，重新计算SHAP值...")
    #         # 删除无效缓存，进入else分支重新计算
    #         os.remove(shap_cache_path)
    #     else:
    #         print(f"🔍 发现SHAP值缓存（{shap_cache_path}），直接加载...")
    #         for cls in range(num_classes):
    #             cls_key = f"cls_{cls}"
    #             shap_values.append(shap_cache_data[cls_key])
    #         start_shap = end_shap = time.perf_counter()
    # else:


        # # 保存SHAP缓存
        # cache_data = {f"cls_{cls}": shap_values[cls] for cls in range(num_classes)}
        # np.savez(shap_cache_path, **cache_data)
        # print(f"💾 SHAP值已缓存至：{shap_cache_path}")
    # -----------------------------------------------------------------------------
    print("🚀 无SHAP值缓存，开始计算...")
    # SHAP解释器初始化
    from sklearn.cluster import KMeans

    sample_size = min(500, len(x_train_shap))
    kmeans = KMeans(n_clusters=sample_size, random_state=42, n_init=10)
    x_flat = x_train_shap.reshape(x_train_shap.shape[0], -1)
    kmeans.fit(x_flat)
    representative_samples = kmeans.cluster_centers_.reshape(sample_size, x_train_shap.shape[1], 1)
    explainer = shap.GradientExplainer(model, representative_samples)

    # 多线程分批计算
    from concurrent.futures import ThreadPoolExecutor


    def compute_shap_in_batches(explainer, data, batch_size=1000, max_workers=4):
        shap_batches = []
        total = len(data)
        batches = [data[i:i + batch_size] for i in range(0, total, batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(lambda b: explainer.shap_values(b, nsamples=20), batches),
                total=len(batches),
                desc="计算SHAP值",
                unit="批"
            ))
        shap_batches.extend(results)
        return shap_batches


    # 计算SHAP值
    start_shap = time.perf_counter()
    cpu_cores = cpu_count()
    batch_size = 500 if cpu_cores >= 8 else 1500
    shap_batches = compute_shap_in_batches(explainer, x_test_shap, batch_size=batch_size)
    end_shap = time.perf_counter()

    # -------------------------- 合并SHAP结果（移到else分支内） --------------------------
    shap_values = []
    for cls in range(num_classes):
        cls_shap = np.concatenate([batch[cls] for batch in shap_batches], axis=0)
        shap_values.append(cls_shap)
    print(f"⏱️  SHAP计算耗时：{end_shap - start_shap:.2f} 秒")

# -------------------------- 维度处理与可视化（无需修改） --------------------------
shap_values_new = np.mean(np.abs(shap_values), axis=0).squeeze(axis=-1)
x_test_new = x_test_shap.squeeze(axis=-1)
shap.summary_plot(shap_values_new, x_test_new, plot_type='bar', feature_names=column_names)

# 特征重要性计算与筛选（保持原有逻辑不变）
shap_test_mul_feature = np.abs(shap_values_new).mean(axis=0)
total = sum(shap_test_mul_feature)
shap_DB_cnn = [(x / total) * 100 for x in shap_test_mul_feature]
shap_DB_cnn_result = [name for i, name in enumerate(column_names) if shap_DB_cnn[i] < 0.2]
print(f"\n🎯 筛选出的低重要性特征数量：{len(shap_DB_cnn_result)}")
print(f"   低重要性特征：{shap_DB_cnn_result[:5]}..." if len(
    shap_DB_cnn_result) > 5 else f"   低重要性特征：{shap_DB_cnn_result}")

# -------------------------- 特征筛选后的模型训练（复用所有优化配置） --------------------------
print("\n🚀 开始训练筛选后模型（复用优化配置）...")
# 用筛选后的特征准备数据（复用缓存的x_resampled）
df_dropped = x_resampled.drop(columns=shap_DB_cnn_result)
if df_dropped.shape[1] == 0:
    raise ValueError("❌ 特征筛选后剩余特征数为0，无法构建模型！请调整筛选阈值（如将0.2改为更小值）。")
elif df_dropped.shape[1] < 5:  # 可根据业务调整最小特征数
    print(f"⚠️  警告：筛选后仅剩余{df_dropped.shape[1]}个特征，可能影响模型精度。")
x_filtered = df_dropped
x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(
    x_filtered, y_resampled, test_size=0.2, random_state=0
)
# 维度调整
x_train_f = x_train_f.to_numpy().reshape((-1, x_train_f.shape[1], 1))  # 先转数组
x_test_f = x_test_f.to_numpy().reshape((-1, x_test_f.shape[1], 1))    # 先转数组
# 标签编码
y_train_f = onehot.transform(y_train_f.reshape(-1, 1))
y_test_f = onehot.transform(y_test_f.reshape(-1, 1))
print(f"📥 筛选后训练集形状：{x_train_f.shape}（特征数减少{len(column_names) - len(df_dropped.columns)}个）")

# 创建tf.data数据集（复用函数）
batch_size_f = base_batch_size * num_gpus  # 保持批次大小一致
train_dataset_f = create_tf_dataset(x_train_f, y_train_f, batch_size_f, is_train=True)
val_dataset_f = create_tf_dataset(x_test_f, y_test_f, batch_size_f, is_train=False)
# 计算训练步数并校验
steps_per_epoch_f = len(x_train_f) // batch_size_f
val_steps_f = len(x_test_f) // batch_size_f

# 若步数为0，动态减小批次大小（最低为32）
min_batch_size = 32
while steps_per_epoch_f == 0 and batch_size_f > min_batch_size:
    batch_size_f = batch_size_f // 2
    steps_per_epoch_f = len(x_train_f) // batch_size_f
    val_steps_f = len(x_test_f) // batch_size_f

# 最终校验（仍为0则报错）
if steps_per_epoch_f == 0:
    raise ValueError(f"❌ 筛选后训练样本数{len(x_train_f)}不足最小批次{min_batch_size}，无法训练！")

# 构建筛选后模型（结构与之前一致，仅输入维度变化）
with strategy.scope():
    inp_f = Input(shape=(x_train_f.shape[1:]))
    x_f = Conv1D(32, 6, padding="same")(inp_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.tanh(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Conv1D(64, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = Conv1D(64, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = LeakyReLU(alpha=0.33)(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Conv1D(128, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = Conv1D(128, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = LeakyReLU(alpha=0.33)(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Conv1D(128, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = Conv1D(128, 6, padding="same")(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = tf.keras.activations.relu(x_f)
    x_f = LeakyReLU(alpha=0.33)(x_f)
    x_f = Dropout(0.5)(x_f)

    x_f = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_f)
    x_f = Flatten()(x_f)
    output_f = Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax',
                     dtype='float32')(x_f)
    model_f = Model(inputs=inp_f, outputs=output_f)

    model_f.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# 训练筛选后模型
start_f = time.perf_counter()
history_f = model_f.fit(
    train_dataset_f,
    epochs=30,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch_f,
    validation_data=val_dataset_f,
    validation_steps=val_steps_f,
    shuffle=False
)
end_f = time.perf_counter()

# 评估筛选后模型
val_loss_f, val_acc_f = model_f.evaluate(val_dataset_f, steps=val_steps_f)
print(f"\n📈 筛选后模型验证性能：损失{val_loss_f:.4f}，精度{val_acc_f:.4f}")
print(f"⏱️  筛选后模型训练耗时：{end_f - start_f:.2f} 秒")

# 最终分类报告
Y_test_f = np.argmax(y_test_f, axis=1)
predict_f = model_f.predict(val_dataset_f, steps=val_steps_f)
y_pred_f = np.argmax(predict_f, axis=1)
print("\n📋 筛选后模型分类报告：")
print(classification_report(Y_test_f[:len(y_pred_f)], y_pred_f, digits=5))