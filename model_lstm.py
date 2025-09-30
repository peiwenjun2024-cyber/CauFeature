import random
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

import shared_globals

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽INFO级及以下日志
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置为默认字体，避免警告
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SVMSMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, LSTM
from tensorflow.keras.regularizers import l2


# 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def data_pre_process(datapath):
    # 读取数据
    df = pd.read_csv(datapath, encoding="gbk")
    shared_globals.data = df
    x, y = df.iloc[:, 0:-1], df.iloc[:, -1]
    shared_globals.feature_data = x
    shared_globals.feature_names = x.columns.tolist()
    shared_globals.target_name = [df.columns[-1]]
    shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name


def load_and_preprocess_data(datapath, filtered_features=None):
    """加载数据并进行预处理（支持特征筛选）"""
    df = pd.read_csv(datapath, encoding="gbk")
    shared_globals.data = df

    # 新增：保留临床意义重要的特征，即使它们被筛选掉
    clinically_important = ['thalach', 'trestbps']  # 临床重要特征

    if filtered_features is None:
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

        # 处理缺失值
        if x.isnull().any().any():
            print(f"检测到缺失值，使用均值填充...")
            x = x.fillna(x.mean())

        # 处理目标变量
        if not pd.api.types.is_numeric_dtype(y):
            print(f"目标变量为非数值型，进行编码")
            y_le = LabelEncoder()
            y = pd.Series(
                y_le.fit_transform(y),
                index=y.index,
                name=y.name
            )

        df = pd.concat([x, y], axis=1)
        shared_globals.data = df

        shared_globals.feature_data = x
        shared_globals.feature_names = x.columns.tolist()
        shared_globals.target_name = [df.columns[-1]]
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num = len(shared_globals.data)

        print(f"数据集形状：{x.shape}")
    else:
        print(f"原始数据形状：{df.shape}")

        # 优化：保留临床重要特征，即使它们不在筛选列表中
        valid_features = [f for f in shared_globals.feature_data.columns if f in filtered_features]
        for feature in clinically_important:
            if feature in shared_globals.feature_data.columns and feature not in valid_features:
                valid_features.append(feature)
                print(f"保留临床重要特征: {feature}")

        if not valid_features:
            raise ValueError("筛选的特征在数据中不存在")

        if filtered_features != valid_features:
            print(f"警告：筛选特征顺序已调整为与原始数据一致")

        x = shared_globals.feature_data[valid_features]
        y = df.iloc[:, -1]

        # 处理非数值型特征
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])

        # 处理缺失值
        if x.isnull().any().any():
            print(f"检测到缺失值，使用均值填充...")
            x = x.fillna(x.mean())

        # 处理目标变量
        if not pd.api.types.is_numeric_dtype(y):
            print(f"目标变量为非数值型，进行编码")
            y_le = LabelEncoder()
            y = pd.Series(
                y_le.fit_transform(y),
                index=y.index,
                name=y.name
            )

        df = pd.concat([x, y], axis=1)
        shared_globals.data = df

        shared_globals.feature_data = x
        shared_globals.feature_names = x.columns.tolist()
        shared_globals.target_name = [df.columns[-1]]
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num = len(shared_globals.data)

        print(f"筛选后数据集形状：{x.shape}")
        print(f"使用筛选后的特征：{valid_features}")

    return x, y


def build_lstm_model(input_shape, num_classes):
    """构建自适应LSTM模型，根据输入特征数量调整结构"""
    inp = Input(shape=input_shape)
    num_features = input_shape[0]  # 获取特征数量

    # 优化：根据特征数量动态调整LSTM单元数
    if num_features <= 5:  # 特征较少时使用较小的单元数
        units1, units2 = 32, 64
    elif num_features <= 10:  # 中等特征数量
        units1, units2 = 48, 96
    else:  # 特征较多时
        units1, units2 = 64, 128

    # 特征提取层
    x = LSTM(
        units=units1,
        return_sequences=True,
        kernel_initializer='he_normal',
        recurrent_dropout=0.2  # 新增：循环层dropout，减少过拟合
    )(inp)
    x = Dropout(0.4)(x)  # 降低dropout率，保留更多特征信息
    x = BatchNormalization()(x)

    # 第二层LSTM
    x = LSTM(
        units=units2,
        kernel_initializer='he_normal',
        recurrent_dropout=0.2
    )(x)
    x = LeakyReLU(alpha=0.33)(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    # 分类层：根据特征数量调整全连接层大小
    dense_units = max(64, int(units2 * 0.75))
    q = Dense(dense_units, kernel_initializer='he_normal')(x)
    q = LeakyReLU(alpha=0.33)(q)
    q = Dropout(0.3)(x)  # 进一步降低dropout率

    # 输出层：降低L2正则化强度
    output = Dense(
        num_classes,
        kernel_regularizer=l2(0.005),  # 正则化强度减半
        activation='softmax'
    )(q)
    model = Model(inputs=inp, outputs=output)

    # 优化：使用学习率调度器而非固定学习率
    optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=150,  # 增加训练轮次
        batch_size=128,  # 减小批大小，适合较少特征
        early_stopping_monitor='val_accuracy',  # 改为监控准确率
        early_stopping_patience=8,  # 增加耐心，允许更多轮次
        early_stopping_min_delta=0.0005,  # 降低最小改善阈值
        early_stopping_restore_best=True
):
    """优化训练策略"""
    # 早停策略
    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=early_stopping_restore_best,
        verbose=1
    )

    # 学习率调整：更平缓的调整策略
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=4,
        factor=0.7,  # 降低学习率衰减因子
        min_lr=0.000001,
        verbose=1
    )

    callbacks = [early_stopping, reduce_lr]

    # 训练模型
    start_time = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
        verbose=1  # 显示训练过程
    )
    end_time = time.perf_counter()

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
    """评估模型性能"""
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
        output_dict=True,
        zero_division=1
    )

    if shared_globals.filtered_features is not None:
        tmp = shared_globals.accuracy
        shared_globals.accuracy = report_dict['accuracy']
        shared_globals.accuracy = 100 * (shared_globals.accuracy - tmp)
    else:
        shared_globals.accuracy = report_dict['accuracy']

    print(classification_report(y_test_true, y_pred, digits=5, zero_division=1))

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


def save_model(model, data_path, model_name):
    """保存模型"""
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    save_filename = f"{dataset_name}_{model_name}.keras"
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, save_filename)

    try:
        model.save(model_path)
        print(f"\nLSTM模型已成功保存至: {os.path.abspath(model_path)}")
        shared_globals.model = model
        return True
    except Exception as e:
        print(f"\nLSTM模型保存失败: {str(e)}")
        return False


def train(data_path, filtered_features):
    import shared_globals
    set_seed(shared_globals.random_seed)

    x, y = load_and_preprocess_data(data_path, filtered_features)

    # 分割数据集
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(
        x, y, test_size=0.2, random_state=shared_globals.random_seed
    )

    # 对训练集进行过采样
    sm = SVMSMOTE(random_state=shared_globals.random_seed)
    x_train, y_train = sm.fit_resample(x_train_raw, y_train_raw)

    # 新增：标准化特征，帮助LSTM更好地学习
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.values)
    x_test = scaler.transform(x_test.values)

    # 调整形状
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # 标签进行独热编码
    onehot = OneHotEncoder(sparse_output=False)
    y_train = onehot.fit_transform(y_train.values.reshape(-1, 1))
    y_test = onehot.transform(y_test.values.reshape(-1, 1))

    # 构建模型
    num_classes = len(np.unique(y))
    model = build_lstm_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    # model.summary()  # 可选：打印模型结构

    # 训练模型
    print("\n开始训练LSTM模型...")
    # 根据特征数量调整训练参数
    if x_train.shape[1] <= 5:
        # 特征较少时增加训练轮次和耐心
        model, history = train_model(model, x_train, y_train, x_test, y_test,
                                     epochs=200, early_stopping_patience=10)
    else:
        model, history = train_model(model, x_train, y_train, x_test, y_test)

    # 评估模型
    print("\n评估LSTM模型性能...")
    test_acc = evaluate_model(model, x_test, y_test, history)

    # 保存模型
    save_model(model, data_path, model_name="lstm_model")

    return model
