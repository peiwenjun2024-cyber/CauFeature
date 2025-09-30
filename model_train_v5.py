#v5版本致力于解决模型训练与特征样本复杂度问题   失败版本
import random
import time
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
from imblearn.over_sampling import SVMSMOTE, ADASYN
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
    df = pd.read_csv(datapath, encoding="gbk")
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


def build_model(input_shape, num_classes,feature_dim,sample_size):
    """根据数据集特征维度和样本大小动态构建1D卷积神经网络模型"""

    # 1. 动态计算卷积核大小（核心调整）
    if feature_dim < 20:  # 低维度特征
        # 卷积核=特征维度的1/3~1/2（至少3，最大5）
        kernel_size = min(5, max(3, feature_dim // 3))
    elif 20 <= feature_dim <= 50:  # 中维度特征
        # 卷积核=特征维度的1/4~1/3（至少5，最大7）
        kernel_size = min(7, max(5, feature_dim // 4))
    else:  # 高维度特征
        # 卷积核=特征维度的1/5~1/4（至少7，最大9）
        kernel_size = min(9, max(7, feature_dim // 5))

        # 小样本额外限制（避免卷积核过大导致过拟合）
    if sample_size < 1000:
        kernel_size = min(kernel_size, 5)  # 小样本最大卷积核不超过5


    # 1. 动态确定模型参数
    # 小样本优先判断（<1k样本）
    if sample_size < 1000:
        conv_filters = [32, 32, 64, 64]  # 长度4  # 非递进式，以32为主
        num_conv_layers = min(4, feature_dim)  # 层数不超过特征维度
        reduction_layers = 1  # 降维次数
    else:
        # 按特征维度判断
        if sample_size < 1000:
            conv_filters = [32, 32, 64]  # 小样本：最多3层
            num_conv_layers = min(3, feature_dim)  # 从4→3层
            reduction_layers = 1
        else:
            if feature_dim < 20:  # 低维度（核心优化）
                conv_filters = [32, 64, 64]  # 最多3层
                num_conv_layers = min(3, feature_dim)  # 从4→3层（适配低维度）
                reduction_layers = 1
            elif 20 <= feature_dim <= 50:  # 中维度
                conv_filters = [32, 64, 128, 128]  # 最多4层
                num_conv_layers = min(4, feature_dim)  # 从6→4层
                reduction_layers = 1
            else:  # 高维度
                conv_filters = [64, 128, 256, 256]  # 最多6层
                num_conv_layers = min(6, feature_dim)  # 从8→6层
                reduction_layers = 2


    # 确保卷积层数量与过滤器数量匹配（截取后长度=num_conv_layers）
    conv_filters = conv_filters[:num_conv_layers]
    print(f"动态调整：卷积核大小={kernel_size}，卷积层数={num_conv_layers}，过滤器={conv_filters}")  # 调试用


    # 2. 构建模型输入
    inp = Input(shape=input_shape)
    x = inp

    # 3. 动态添加卷积特征提取层
    for i in range(num_conv_layers):
        # 添加卷积层
        x = Conv1D(
            filters=conv_filters[i],
            kernel_size=kernel_size,
            padding="same",
            activation='relu' if i > 0 else 'tanh'  # 第一层用tanh，其余用relu
        )(x)

        # 激活函数（LeakyReLU用于非第一层）
        if i > 0:
            x = LeakyReLU(alpha=0.33)(x)

        #  dropout和批归一化
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

        # 降维策略（高维度时增加降维）
        if (i + 1) % (num_conv_layers // reduction_layers) == 0 and reduction_layers > 0:
            x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)

    # 4. 分类层（根据特征复杂度调整）
    x = Flatten()(x)
    # 动态调整全连接层神经元数量
    dense_units = 256 if feature_dim > 50 else 128
    x = Dense(dense_units, kernel_regularizer=l2(0.001))(x)  # 新增L2正则化
    x = LeakyReLU(alpha=0.33)(x)
    x = Dropout(0.3)(x)  # 降低Dropout比例，减少信息丢失

    # 输出层
    output = Dense(num_classes, kernel_regularizer=l2(0.01), activation='softmax')(x)
    model = Model(inputs=inp, outputs=output)

    # 5. 编译模型（学习率根据样本量动态调整）
    learning_rate = 0.0005 if sample_size < 1000 else 0.001


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],

    )

    return model


import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=100,
        class_weight=None,  # 新增：接收权重字典

        # 早停策略参数（新增灵活配置）
        early_stopping_monitor='val_loss',  # 监控指标（如val_accuracy）
        early_stopping_patience=10,  # 多少轮无改善后停止
        early_stopping_min_delta=0.001,  # 最小改善阈值（小于此值视为无改善）
        early_stopping_restore_best=True  # 是否恢复最佳权重
):
    """训练模型并返回训练历史和耗时，不依赖全局变量"""
    # 动态计算批处理大小（至少8，最多256）
    batch_size = min(256, max(8, len(x_train) // 5))  # 确保至少有5个批次
    print(f"自动调整批处理大小为: {batch_size}")

    """训练模型并记录训练历史（含早停策略和学习率调整）"""
    # 定义早停策略
    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,  # 监控的指标（验证集损失/准确率）
        patience=early_stopping_patience,  # 容忍多少轮无改善
        min_delta=early_stopping_min_delta,  # 最小改善量（避免微小波动）
        restore_best_weights=early_stopping_restore_best,  # 恢复到最佳轮次的权重
        verbose=1  # 打印早停信息（如“Epoch 00020: early stopping”）
    )

    # 学习率调整策略（保持不变）
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        min_lr=0.000001,
        verbose=0
    )



    # 训练模型
    start_time = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,  # 使用传入的字典格式权重
        validation_data=(x_test, y_test),
        shuffle=True,
        # 整合回调函数（包含早停）
        callbacks=[early_stopping, reduce_lr],
        verbose=0,  # 添加此行，设置为0关闭打印

    )
    end_time = time.perf_counter()





    if shared_globals.filtered_features is not None:
        tmp=shared_globals.running_time
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
        zero_division = 1,
        output_dict = True
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


def  train(data_path,filtered_features):
    # 文件路径处理（兼容不同操作系统）
    import shared_globals
    set_seed(shared_globals.random_seed)

    x, y = load_and_preprocess_data(data_path,filtered_features)

    # 分割数据集（先分割，后过采样，避免数据泄露）
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(x, y, test_size=0.2, random_state=shared_globals.random_seed)

    # 新增：计算类别权重（基于原始训练集，过采样前）
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_raw)  # 获取所有类别
    class_weights = compute_class_weight(
        class_weight='balanced',  # 使用"balanced"策略
        classes=classes,
        y=y_train_raw  # 传入原始训练标签（非独热编码）
    )
    shared_globals.class_weights=class_weights
    class_weight_dict = {i: class_weights[i] for i in range(len(classes))}  # 转换为字典
    print(f"计算的类别权重: {class_weight_dict}")  # 调试用


    # 替换SVMSMOTE为ADASYN
    sm = ADASYN(random_state=shared_globals.random_seed, sampling_strategy='minority')  # 只对少数类过采样
    x_train, y_train = sm.fit_resample(x_train_raw, y_train_raw)
    print(f"过采样后类别分布:\n{pd.Series(y_train).value_counts()}")  # 验证采样效果

    # print(f"过采样后训练集形状: {x_train.shape}")
    # print(f"过采样后类别分布:\n{pd.Series(y_train).value_counts()}")

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
    if shared_globals.filtered_features is not None:
        model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes,
                            feature_dim=len(shared_globals.filtered_features), sample_size=shared_globals.sample_num)
    else:
        model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes,
                            feature_dim=len(shared_globals.feature_names), sample_size=shared_globals.sample_num)


    # 训练模型
    # print("\n开始训练模型...")
    model, history = train_model(model, x_train, y_train, x_test, y_test,class_weight=class_weight_dict)

    # 评估模型
    # print("\n评估模型性能...")
    test_acc = evaluate_model(model, x_test, y_test, history)

    return model






