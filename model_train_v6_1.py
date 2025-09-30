import random
import time

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

import shared_globals

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽INFO级及以下日志（只保留ERROR）
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置为默认字体，避免警告
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

# 使用英文标签
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SVMSMOTE, SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 设置随机种子，确保结果可复现
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # 禁用oneDNN加速以减少数值波动
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu

def data_pre_process(datapath):
    # 读取数据
    df = pd.read_csv(datapath, encoding="gbk")
    shared_globals.data = df
    x, y = df.iloc[:, 0:-1], df.iloc[:, -1]
    shared_globals.feature_data = x  # 保存预处理后的特征数据（x为DataFrame）
    shared_globals.feature_names = x.columns.tolist()  # 特征名称
    shared_globals.target_name = [df.columns[-1]]  # 保持列表格式，与全局变量定义一致
    shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name

def load_and_preprocess_data(datapath, filtered_features=None):
    """加载数据并进行预处理（支持特征筛选）"""
    # 读取数据
    df = pd.read_csv(datapath) #delimiter=';',, encoding="utf-8"
    shared_globals.data=df





    if filtered_features is None:
        # 不筛选特征的逻辑（原第一个函数功能）
        print(f"数据基本信息：")
        df.info()
        x, y = df.iloc[:, 0:-1], df.iloc[:, -1]

        # 处理非数值型特征
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])

        # 新增：处理缺失值（关键修复）
        if x.isnull().any().any():
            print(f"检测到缺失值，使用均值填充...")
            # 对数值型特征用均值填充，非数值型用众数（此处已编码为数值，直接用均值）
            x = x.fillna(x.mean())

        # 处理目标变量（若为非数值型则编码，关键：保持为Series）
        if not pd.api.types.is_numeric_dtype(y):
            print(f"目标变量为非数值型，进行编码")
            y_le = LabelEncoder()
            # 用 Series 包裹编码结果，保留索引和名称
            y = pd.Series(
                y_le.fit_transform(y),
                index=y.index,
                name=y.name
            )

        # 合并特征与编码后的目标变量
        df = pd.concat([x, y], axis=1)

        # 关键：将编码后的完整数据集赋值给全局变量
        shared_globals.data = df

        shared_globals.feature_data = x  # 保存预处理后的特征数据（x为DataFrame）
        shared_globals.feature_names = x.columns.tolist()  # 特征名称
        shared_globals.target_name = [df.columns[-1]]  # 保持列表格式，与全局变量定义一致
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num=len(shared_globals.data)

        print(f"数据集形状：{x.shape}")
    else:
        # 筛选特征的逻辑（原第二个函数功能）
        print(f"原始数据形状：{df.shape}")
        # 筛选有效特征

        valid_features = [f for f in shared_globals.feature_data.columns if f in filtered_features]
        if not valid_features:
            raise ValueError("筛选的特征在数据中不存在")

        # 检查filtered_features是否与feature_data的顺序一致
        if filtered_features != valid_features:
            print(f"警告：筛选特征顺序已调整为与原始数据一致")
            #print(f"警告：筛选特征顺序已调整为与原始数据一致，原顺序：{filtered_features}，新顺序：{valid_features}")

        x = shared_globals.feature_data[valid_features]
        y = df.iloc[:, -1]

        # 处理非数值型特征
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])

        # 新增：处理缺失值（关键修复）
        if x.isnull().any().any():
            print(f"检测到缺失值，使用均值填充...")
            x = x.fillna(x.mean())

        # 处理目标变量（若为非数值型则编码，关键：保持为Series）
        if not pd.api.types.is_numeric_dtype(y):
            print(f"目标变量为非数值型，进行编码")
            y_le = LabelEncoder()
            # 用 Series 包裹编码结果，保留索引和名称
            y = pd.Series(
                y_le.fit_transform(y),
                index=y.index,
                name=y.name
            )

        # 合并特征与编码后的目标变量
        df = pd.concat([x, y], axis=1)

        # # 合并编码后的特征与目标变量，形成完整数据集
        # df = pd.concat([x, y], axis=1)  # 按列合并（特征+目标）

        # 关键：将编码后的完整数据集赋值给全局变量
        shared_globals.data = df

        shared_globals.feature_data = x  # 保存预处理后的特征数据（x为DataFrame）
        shared_globals.feature_names = x.columns.tolist()  # 特征名称
        shared_globals.target_name = [df.columns[-1]]  # 保持列表格式，与全局变量定义一致
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num = len(shared_globals.data)

        print(f"筛选后数据集形状：{x.shape}")
        print(f"使用筛选后的特征：{valid_features}")

    # 共用的类别分布打印
    #print(f"类别分布：\n{y.value_counts()}")
    return x, y

#二分类成功
# def smote_enn_resample(x_train, y_train, random_state):
#     # 1. 配置SMOTE：温和过采样少数类（避免生成过多噪音样本）
#     smote = SMOTE(
#         sampling_strategy=0.9,  # 少数类:多数类 = 9:10（接近平衡即可）
#         k_neighbors=5,  # 减少近邻数，避免过度生成不合理样本
#         random_state=random_state
#     )
#
#     # 2. 配置ENN：减少欠采样强度（保留更多多数类样本）
#     enn = EditedNearestNeighbours(
#         n_neighbors=3,  # 减少近邻数，降低移除阈值
#         kind_sel='mode'  # 基于多数投票判断噪音（更保守）
#     )
#
#     # 3. 组合SMOTE+ENN
#     smote_enn = SMOTEENN(
#         sampling_strategy="auto",  # 自动调整为合理比例
#         random_state=random_state,
#         smote=SMOTE(random_state=random_state)  # 可进一步配置SMOTE参数
#     )
#
#     # 4. 执行采样
#     x_resampled, y_resampled = smote_enn.fit_resample(x_train, y_train)
#
#     # 5. 采样后检查，若仍不平衡则补充调整
#     counts = pd.Series(y_resampled).value_counts()
#     print(f"SMOTE+ENN采样后类别分布:\n{counts}")
#
#     # 若两类差异仍大于10%，用SMOTE再微调
#     if abs(counts[0] - counts[1]) / max(counts) > 0.1:
#         print("采样后仍不平衡，进行二次微调...")
#         adjust_smote = SMOTE(
#             sampling_strategy='minority',  # 只调整少数类
#             k_neighbors=3,
#             random_state=random_state
#         )
#         x_resampled, y_resampled = adjust_smote.fit_resample(x_resampled, y_resampled)
#         print(f"二次调整后类别分布:\n{pd.Series(y_resampled).value_counts()}")
#
#     return x_resampled, y_resampled

# 多分类尝试
# 1. 修改smote_enn_resample函数：处理极小众类别
def smote_enn_resample(x_train, y_train, random_state):
    # 转换为DataFrame便于处理（保留索引，避免样本错位）
    train_df = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    target_col = y_train.name  # 目标变量列名

    # 步骤1：分析类别分布，筛选有效类别（样本数≥2）
    class_counts = train_df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index.tolist()  # 有效类别（可采样）
    rare_classes = class_counts[class_counts == 1].index.tolist()  # 极小众类别（仅保留原样本）
    print(f"有效类别数（样本数≥2）: {len(valid_classes)}，极小众类别数（样本数=1）: {len(rare_classes)}")

    # 步骤2：拆分有效类别样本和极小众类别样本
    # 有效类别样本（用于SMOTE+ENN采样）
    valid_samples = train_df[train_df[target_col].isin(valid_classes)]
    valid_x = valid_samples.drop(columns=[target_col])
    valid_y = valid_samples[target_col]
    # 极小众类别样本（直接保留，不参与采样）
    rare_samples = train_df[train_df[target_col].isin(rare_classes)]
    rare_x = rare_samples.drop(columns=[target_col])
    rare_y = rare_samples[target_col]

    # 步骤3：若有效类别样本为空（全部是极小众类别），直接返回原样本（避免后续报错）
    if len(valid_samples) == 0:
        print("警告：无有效类别（全部类别仅1个样本），不进行过采样，直接返回原训练集")
        return x_train.values, y_train.values

    # 步骤4：动态调整有效类别的k_neighbors（确保≥1且≤最小有效类别样本数-1）
    min_valid_samples = class_counts[valid_classes].min()
    k_neighbors = min(3, max(1, min_valid_samples - 1))  # 缩小k_neighbors范围，降低采样风险
    enn_neighbors = min(2, max(1, min_valid_samples - 1))
    print(f"有效类别最小样本数: {min_valid_samples}，调整k_neighbors={k_neighbors}，enn_neighbors={enn_neighbors}")

    # 步骤5：配置SMOTE（仅对有效类别进行采样，避免处理极小众类别）
    smote = SMOTE(
        sampling_strategy='auto',  # 自动平衡有效类别中的少数类
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    # 配置ENN（仅对有效类别样本去噪）
    enn = EditedNearestNeighbours(
        n_neighbors=enn_neighbors,
        kind_sel='mode'
    )
    smote_enn = SMOTEENN(
        sampling_strategy="auto",
        random_state=random_state,
        smote=smote,
        enn=enn
    )

    # 步骤6：对有效类别样本进行SMOTE+ENN采样
    valid_x_resampled, valid_y_resampled = smote_enn.fit_resample(valid_x, valid_y)

    # 步骤7：合并采样后的有效类别样本和原极小众类别样本
    # 采样后的有效样本（转换为DataFrame）
    valid_resampled_df = pd.DataFrame(valid_x_resampled, columns=valid_x.columns)
    valid_resampled_df[target_col] = valid_y_resampled
    # 原极小众样本（直接拼接）
    rare_df = pd.DataFrame(rare_x.values, columns=rare_x.columns)
    rare_df[target_col] = rare_y.values
    # 合并所有样本
    final_train_df = pd.concat([valid_resampled_df, rare_df], ignore_index=True)

    # 步骤8：输出最终分布（验证效果）
    final_counts = final_train_df[target_col].value_counts().sort_index()
    print(f"最终训练集类别分布（前10个类别）:\n{final_counts.head(10)}")
    print(f"最终训练集总样本数: {len(final_train_df)}")

    # 返回numpy数组（适配后续模型输入）
    return final_train_df.drop(columns=[target_col]).values, final_train_df[target_col].values

"""根据特征维度和样本量动态构建1D卷积神经网络模型（新增动态优化器/学习率）"""
def build_model(input_shape, num_classes, sample_num):
    # 特征维度（序列长度）
    D = input_shape[0]
    print(f"根据特征维度 {D} 和样本量 {sample_num} 调整模型参数...")

    # 1. 原有参数调整（卷积核、滤波器等）保持不变
    if D <= 20:
        kernel_size = 3
    elif D <= 100:
        kernel_size = 6
    else:
        kernel_size = 9

    if sample_num <= 10000:  # 小样本
        filters = [16, 32, 64]
        dense_units = 64
        dropout_rate = 0.6
        l2_reg = 0.05
        conv_layers = 2
    elif sample_num <= 100000:  # 中样本
        filters = [32, 64, 128]
        dense_units = 128
        dropout_rate = 0.5
        l2_reg = 0.01
        conv_layers = 3
    else:  # 大样本
        filters = [64, 128, 256]
        dense_units = 256
        dropout_rate = 0.4
        l2_reg = 0.001
        conv_layers = 3

    use_pooling = D > 50

    # 2. 动态设置初始学习率（基于样本量）
    if sample_num <= 10000:
        initial_lr = 0.0005  # 小样本用较小学习率，避免过拟合
    elif sample_num <= 100000:
        initial_lr = 0.001   # 中样本用默认学习率
    else:
        initial_lr = 0.002   # 大样本用较大学习率，加速收敛

    # 3. 动态选择优化器（基于样本量和特征维度）
    if sample_num <= 100000:
        # 中小样本：Adam/Adamax（自适应学习率，适合数据量有限场景）
        if D > 100:  # 高维度特征用Adamax更稳定
            optimizer = keras.optimizers.Adamax(learning_rate=initial_lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    else:
        # 大样本：SGD+动量（收敛更稳健，适合大数据量）
        optimizer = keras.optimizers.SGD(
            learning_rate=initial_lr,
            momentum=0.9,  # 动量加速收敛
            nesterov=True  # 启用Nesterov动量，增强稳定性
        )

    # 4. 模型结构（保持不变）
    inp = Input(shape=input_shape)
    x = Conv1D(filters[0], kernel_size, padding="same", activation='tanh')(inp)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    if use_pooling:
        x = MaxPooling1D(pool_size=2, strides=2)(x)

    if conv_layers >= 2:
        x = Conv1D(filters[1], kernel_size, strides=1, padding='same', activation='relu')(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        if use_pooling and D > 100:
            x = MaxPooling1D(pool_size=2, strides=2)(x)

    if conv_layers >= 3 and (sample_num > 10000 or D > 30):
        x = Conv1D(filters[2], kernel_size, strides=1, padding='same', activation='relu')(x)
        x = LeakyReLU(alpha=0.33)(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)

    x = Flatten()(x)
    q = Dense(dense_units)(x)
    q = LeakyReLU(alpha=0.33)(q)
    q = Dropout(dropout_rate)(q)

    output = Dense(num_classes, kernel_regularizer=l2(l2_reg), activation='softmax')(q)
    model = Model(inputs=inp, outputs=output)

    # 5. 编译模型（使用动态优化器）
    model.compile(
        optimizer=optimizer,  # 动态选择的优化器
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


"""训练模型并记录训练历史（优化学习率调整策略）"""
def train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        sample_num,
        epochs=100,
        early_stopping_monitor='val_loss',
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_restore_best=True
):
    # 1. 原有batch_size调整（保持不变）
    if sample_num <= 10000:
        batch_size = 64
    elif sample_num <= 100000:
        batch_size = 256
    else:
        batch_size = 512
    print(f"根据样本量 {sample_num} 使用batch_size: {batch_size}")

    # 2. 早停策略（保持不变）
    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=early_stopping_restore_best,
        verbose=1
    )

    # 3. 动态学习率衰减（基于初始学习率和样本量）
    initial_lr = model.optimizer.learning_rate.numpy()  # 获取模型初始学习率
    if initial_lr <= 0.0005:  # 小初始学习率：减缓衰减
        lr_factor = 0.8  # 衰减因子（较大，衰减慢）
        lr_patience = 4   # 耐心值（允许更多轮次无改善）
    elif initial_lr <= 0.001:  # 中等初始学习率：标准衰减
        lr_factor = 0.5
        lr_patience = 3
    else:  # 大初始学习率：加速衰减
        lr_factor = 0.3
        lr_patience = 2

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=lr_patience,  # 动态耐心值
        factor=lr_factor,      # 动态衰减因子
        min_lr=initial_lr * 0.001,  # 最小学习率（基于初始值）
        verbose=1  # 打印学习率调整信息
    )

    # 4. 训练模型（保持不变）
    callbacks = [early_stopping, reduce_lr]
    start_time = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
        verbose=0
    )
    end_time = time.perf_counter()

    # 5. 训练时间记录（保持不变）
    if shared_globals.filtered_features is not None:
        tmp = shared_globals.running_time
        shared_globals.running_time = end_time - start_time
        print(f"模型训练耗时: {shared_globals.running_time:.2f}秒")
        shared_globals.running_time = tmp - shared_globals.running_time
    else:
        shared_globals.running_time = end_time - start_time
        print(f"模型训练耗时: {shared_globals.running_time:.2f}秒")

    return model, history



def evaluate_model(model, x_test, y_test, history):
    """评估模型性能并可视化结果"""
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")

    # 预测并生成分类报告
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_true = np.argmax(y_test, axis=1)

    print("\n分类报告:")
    report_dict = classification_report(
        y_test_true,
        y_pred,
        digits=5,
        output_dict=True , # 输出为字典格式
        zero_division = 1

    )

    if shared_globals.filtered_features is not None:
        tmp=shared_globals.accuracy
        shared_globals.accuracy = report_dict['accuracy']
        shared_globals.accuracy = 100* (shared_globals.accuracy - tmp)


    else:
        shared_globals.accuracy = report_dict['accuracy']


    print(classification_report(y_test_true, y_pred, digits=5,zero_division=1))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 绘制训练历史
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return test_acc

def save_model(model,data_path, model_name):
    """
    根据数据集名称和模型名称自动生成保存路径并保存模型

    参数:
        model: 要保存的Keras模型实例
        data_path: 数据集文件的路径（用于提取数据集名称）
        model_name: 模型名称（默认"cnn_model"）
    """
    # 从数据集路径中提取文件名（不含扩展名）
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # 生成保存文件名：数据集名称_模型名称.keras
    save_filename = f"{dataset_name}_{model_name}.keras"

    # 保存目录默认为"saved_models"（可根据需要修改）
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 拼接完整保存路径
    model_path = os.path.join(save_dir, save_filename)



    try:
        model.save(model_path)
        print(f"\n模型已成功保存至: {os.path.abspath(model_path)}")
        shared_globals.model = model  # 保存到全局变量
        return True
    except Exception as e:
        print(f"\n模型保存失败: {str(e)}")
        return False


# 2. 修改train函数：启用特征标准化（关键！修复SMOTE近邻计算失真问题）
def train(data_path, filtered_features):
    import shared_globals
    set_seed(shared_globals.random_seed)
    x, y = load_and_preprocess_data(data_path, filtered_features)

    # 分割数据集（先分割，后过采样）
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(
        x, y, test_size=0.2, random_state=shared_globals.random_seed,
        stratify=y if len(y.value_counts()) <= 10 else None  # 类别太多时不使用stratify，避免分割失败
    )

    # 新增：特征标准化（SMOTE近邻计算依赖特征尺度，必须启用）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # 训练集：拟合+转换（仅用训练集数据，避免数据泄露）
    x_train_raw_scaled = scaler.fit_transform(x_train_raw.values)
    # 测试集：仅转换（用训练集的scaler参数）
    x_test_scaled = scaler.transform(x_test.values)

    # 替换为修复后的SMOTE+ENN采样（使用标准化后的训练集）
    print("采样前训练集类别分布（前10个类别）：")
    print(pd.Series(y_train_raw).value_counts().head(10))
    print("\n应用SMOTE+ENN混合采样（过滤极小众类别）...")
    x_train, y_train = smote_enn_resample(
        pd.DataFrame(x_train_raw_scaled, columns=x_train_raw.columns),  # 标准化后的训练集特征
        y_train_raw,
        random_state=shared_globals.random_seed
    )

    # 数据格式转换（适配CNN输入：添加通道维度）
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test_scaled.reshape((x_test_scaled.shape[0], x_test_scaled.shape[1], 1))  # 使用标准化后的测试集

    # 标签独热编码（注意：若目标变量类别数过多，独热编码会导致维度爆炸，后续需考虑分箱）
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # 忽略测试集中的新类别
    y_train = onehot.fit_transform(y_train.reshape(-1, 1))
    y_test = onehot.transform(y_test.values.reshape(-1, 1))

    # 构建模型（若类别数过多，需减少dense层单元数，避免过拟合）
    num_classes = y_train.shape[1]
    model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes, sample_num=shared_globals.sample_num)

    # 训练模型（若类别数过多，可增加早停patience，给模型更多收敛时间）
    model, history = train_model(
        model, x_train, y_train, x_test, y_test,
        sample_num=shared_globals.sample_num,
        early_stopping_patience=15  # 延长早停耐心值
    )

    # 评估模型
    test_acc = evaluate_model(model, x_test, y_test, history)
    return model






