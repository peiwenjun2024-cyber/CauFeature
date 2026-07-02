import random
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
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

def _encode_target(y):
    """Return a numeric Series for labels, including bool labels for SciPy/imblearn compatibility."""
    if pd.api.types.is_bool_dtype(y):
        return pd.Series(y.astype(np.int64), index=y.index, name=y.name)
    if not pd.api.types.is_numeric_dtype(y):
        y_le = LabelEncoder()
        return pd.Series(
            y_le.fit_transform(y),
            index=y.index,
            name=y.name
        )
    return y

def load_and_preprocess_data(datapath, filtered_features=None):
    """加载数据并进行预处理（支持特征筛选）"""

    from scipy.io import arff  # 或使用liac-arff

    # 根据文件扩展名判断格式
    if datapath.endswith('.arff'):
        # 读取ARFF文件
        data, meta = arff.loadarff(datapath)
        df = pd.DataFrame(data)
        # 解码字符串
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.decode('utf-8')
    else:
        # 读取数据
        df = pd.read_csv(datapath, encoding="utf-8")  # delimiter=';',
    #df = df.sample(frac=0.1, random_state=shared_globals.random_seed)
    shared_globals.data=df





    if filtered_features is None:
        # 不筛选特征的逻辑（原第一个函数功能）
        print(f"Basic Information of Data：")
        df.info()
        x, y = df.iloc[:, 0:-1], df.iloc[:, -1]

        # 处理非数值型特征
        print(f"In data preprocessing...")
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            # print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])

        # 新增：处理缺失值（关键修复）
        if x.isnull().any().any():
            # print(f"检测到缺失值，使用均值填充...")
            # 对数值型特征用均值填充，非数值型用众数（此处已编码为数值，直接用均值）
            x = x.fillna(x.mean())

        y = _encode_target(y)

        # 合并特征与编码后的目标变量
        df = pd.concat([x, y], axis=1)

        # 关键：将编码后的完整数据集赋值给全局变量
        shared_globals.data = df

        shared_globals.feature_data = x  # 保存预处理后的特征数据（x为DataFrame）
        shared_globals.feature_names = x.columns.tolist()  # 特征名称
        shared_globals.target_name = [df.columns[-1]]  # 保持列表格式，与全局变量定义一致
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num=len(shared_globals.data)

        print(f"Original data shape：{x.shape}")
    else:
        # 筛选特征的逻辑（原第二个函数功能）
        print(f"Original data shape：{df.shape}")
        # 筛选有效特征

        valid_features = [f for f in shared_globals.feature_data.columns if f in filtered_features]
        if not valid_features:
            raise ValueError("The selected features do not exist in the data.")

        # 检查filtered_features是否与feature_data的顺序一致
        if filtered_features != valid_features:
            print(f"Warning: The order of filtered features has been adjusted to be consistent with the original data")
            #print(f"警告：筛选特征顺序已调整为与原始数据一致，原顺序：{filtered_features}，新顺序：{valid_features}")

        x = shared_globals.feature_data[valid_features]
        y = df.iloc[:, -1]

        # 处理非数值型特征
        print(f'In data precessing...')
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            # print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])

        # 新增：处理缺失值（关键修复）
        if x.isnull().any().any():
            # print(f"检测到缺失值，使用均值填充...")
            x = x.fillna(x.mean())

        y = _encode_target(y)

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

        print(f"data shape after CauFeature：{x.shape}")
        print(f"Features after CauFeature：{valid_features}")

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
def smote_enn_resample(x_train, y_train, random_state):
    # 1. 配置SMOTE：多分类场景下自动平衡各类别
    smote = SMOTE(
        sampling_strategy='auto',  # 多分类默认策略：将少数类采样至多数类的一半
        k_neighbors=5,
        random_state=random_state
    )

    # 2. 配置ENN：删除不支持的allow_degenerate参数
    enn = EditedNearestNeighbours(
        n_neighbors=3,
        kind_sel='mode'  # 多分类中使用多数投票判断噪音
        # 移除allow_degenerate=True（该参数不存在于EditedNearestNeighbours）
    )

    # 3. 组合SMOTE+ENN（多分类兼容）
    smote_enn = SMOTEENN(
        sampling_strategy="auto",
        random_state=random_state,
        smote=smote,
        enn=enn
    )

    # 4. 执行采样
    x_resampled, y_resampled = smote_enn.fit_resample(x_train, y_train)

    # 5. 采样后检查类别分布
    counts = pd.Series(y_resampled).value_counts().sort_index()
    # print(f"SMOTE+ENN采样后类别分布:\n{counts}")

    # 6. 多分类平衡判断：计算最大最小类别比例
    max_count = counts.max()
    min_count = counts.min()
    balance_ratio = min_count / max_count  # 最小类别占最大类别的比例

    # 若最不平衡比例低于0.8，进行二次微调（多分类标准）
    if balance_ratio < 0.8:
        # print(f"类别平衡比例为{balance_ratio:.2f}，进行二次微调...")
        # 针对少数类进行补充采样
        adjust_smote = SMOTE(
            sampling_strategy={
                cls: max_count if cnt < max_count * 0.8 else cnt
                for cls, cnt in counts.items()
            },  # 只调整比例低于80%的类别
            k_neighbors=3,
            random_state=random_state
        )
        x_resampled, y_resampled = adjust_smote.fit_resample(x_resampled, y_resampled)
        # 输出调整后分布
        adjusted_counts = pd.Series(y_resampled).value_counts().sort_index()
        # print(f"二次调整后类别分布:\n{adjusted_counts}")

    return x_resampled, y_resampled

"""根据特征维度和样本量动态构建1D卷积神经网络模型（新增动态优化器/学习率）"""
def build_model(input_shape, num_classes, sample_num):
    # 特征维度（序列长度）
    D = input_shape[0]
    print(f"Adjust the model parameters according to dimension {D} and sample size {sample_num} ...")

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
    # print(f"根据样本量 {sample_num} 使用batch_size: {batch_size}")

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
        # print(f"模型训练耗时: {shared_globals.running_time:.2f}秒")
        shared_globals.running_time = tmp - shared_globals.running_time
    else:
        shared_globals.running_time = end_time - start_time
        # print(f"模型训练耗时: {shared_globals.running_time:.2f}秒")

    return model, history



def evaluate_model(model, x_test, y_test, history):
    """评估模型性能并可视化结果"""
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    # print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")

    # 预测并生成分类报告
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_true = np.argmax(y_test, axis=1)


    report_dict = classification_report(
        y_test_true,
        y_pred,
        digits=5,
        output_dict=True , # 输出为字典格式
        zero_division = 1

    )



    if shared_globals.filtered_features is not None:

        shared_globals.afterreport = report_dict

        tmp = shared_globals.accuracy
        print(f"original accurancy:{tmp}")
        tmpp = shared_globals.f_score
        print(f"original F-measure:{tmpp}")

        shared_globals.accuracy = report_dict['accuracy']
        shared_globals.f_score = report_dict['macro avg']['f1-score']

        shared_globals.accuracy = 100 * (shared_globals.accuracy - tmp)
        shared_globals.f_score = 100 * (float(shared_globals.f_score) - float(tmpp))


    else:
        shared_globals.originalreport = report_dict

        shared_globals.accuracy = report_dict['accuracy']
        shared_globals.f_score = report_dict['macro avg']['f1-score']

    print("\nClassification Report:")
    print(classification_report(y_test_true, y_pred, digits=5,zero_division=1))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_true, y_pred)
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
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
    if shared_globals.filtered_features is not None:
        save_dir="after_models"
    else:
        save_dir = "former_models"
    os.makedirs(save_dir, exist_ok=True)

    # 拼接完整保存路径
    model_path = os.path.join(save_dir, save_filename)



    try:
        model.save(model_path)
        # print(f"\n模型已成功保存至: {os.path.abspath(model_path)}")
        shared_globals.model = model  # 保存到全局变量
        return True
    except Exception as e:
        # print(f"\n模型保存失败: {str(e)}")
        return False


def  train(data_path,filtered_features):
    # 文件路径处理（兼容不同操作系统）
    import shared_globals
    set_seed(shared_globals.random_seed)

    x, y = load_and_preprocess_data(data_path,filtered_features)

    # 分割数据集（先分割，后过采样，避免数据泄露）
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(x, y, test_size=0.2, random_state=shared_globals.random_seed)

    # 替换原有SVMSMOTE为SMOTE+ENN混合采样
    print("Class distribution of the training set before sampling：")
    print(pd.Series(y_train_raw).value_counts(normalize=True))
    print("\nApply SMOTE ENN hybrid sampling...")

    x_train, y_train = smote_enn_resample(
        x_train_raw, y_train_raw,
        random_state=shared_globals.random_seed
    )

    print(f"Shape of the training set after oversampling: {x_train.shape}")
    print(f"Class distribution after oversampling:\n{pd.Series(y_train).value_counts()}")

    # 数据格式转换，适配CNN输入
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train.values)
    # x_test = scaler.transform(x_test.values)

    x_train = x_train.values  # 转换为numpy数组（不进行标准化）
    x_test = x_test.values  # 转换为numpy数组（不进行标准化）

    # 添加这两行以增加通道维度
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # 形状变为 (样本数, 13, 1)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))  # 形状变为 (样本数, 13, 1)
    #x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    #x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    # 标签进行独热编码
    onehot = OneHotEncoder(sparse_output=False)
    y_train = onehot.fit_transform(y_train.values.reshape(-1, 1))
    y_test = onehot.transform(y_test.values.reshape(-1, 1))

    # 构建模型
    # print("\n构建1D卷积神经网络模型...")
    num_classes = len(np.unique(y))
    model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes, sample_num=shared_globals.sample_num)
    #model.summary()

    # 训练模型
    # print("\n开始训练模型...")
    model, history = train_model(model, x_train, y_train, x_test, y_test,shared_globals.sample_num)

    # 评估模型
    # print("\n评估模型性能...")
    test_acc = evaluate_model(model, x_test, y_test, history)

    return model






