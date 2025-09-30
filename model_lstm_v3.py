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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM  # 移除LeakyReLU（适配cuDNN）
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
    # 新增：保存原始特征统计信息（用于后续特征扰动的合理幅度计算）
    shared_globals.feature_stats = {
        'mean': x.mean().to_dict(),
        'std': x.std().to_dict(),
        'min': x.min().to_dict(),
        'max': x.max().to_dict()
    }
    print(f"原始特征统计信息已保存（均值/标准差/最值），用于特征扰动模块")


def load_and_preprocess_data(datapath, filtered_features=None):
    """加载数据并进行预处理（支持特征筛选）"""
    df = pd.read_csv(datapath, encoding="gbk")
    shared_globals.data = df

    # 1. 保留临床重要特征（确保关键医学特征不被筛除）
    clinically_important = ['thalach', 'trestbps', 'chol']  # 新增chol（胆固醇，临床关键）
    # 输出特征原始统计信息（辅助验证扰动幅度合理性）
    print(f"\n关键特征原始统计（用于特征扰动参考）：")
    for feat in clinically_important:
        if feat in df.columns:
            print(f"  {feat}: 均值={df[feat].mean():.2f}, 标准差={df[feat].std():.2f}, 范围=[{df[feat].min()}, {df[feat].max()}]")

    if filtered_features is None:
        print(f"\n数据基本信息：")
        df.info()
        x, y = df.iloc[:, 0:-1], df.iloc[:, -1]

        # 处理非数值型特征
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"发现非数值型特征，将进行编码：{non_numeric_cols}")
            le = LabelEncoder()
            for col in non_numeric_cols:
                x[col] = le.fit_transform(x[col])
                # 保存编码器到全局变量（后续扰动时保持编码一致性）
                shared_globals.label_encoders[col] = le

        # 处理缺失值
        if x.isnull().any().any():
            print(f"检测到缺失值，使用均值填充...")
            x = x.fillna(x.mean())

        # ========== 修复：统一初始化target_encoder ==========
        # 处理目标变量（无论是否为数值型，都初始化LabelEncoder）
        print(f"目标变量类型: {y.dtype}, 名称: {y.name}")
        y_le = LabelEncoder()
        # 对y.values编码（避免Series索引干扰）
        y_encoded = y_le.fit_transform(y.values)
        # 重构Series，保留原始索引和名称
        y = pd.Series(
            y_encoded,
            index=y.index,
            name=y.name
        )
        # 强制保存encoder到全局变量
        shared_globals.target_encoder = y_le
        # 打印类别映射关系，便于验证
        class_mapping = dict(zip(y_le.classes_, y_le.transform(y_le.classes_)))
        print(f"目标变量编码完成，原始类别→编码值映射: {class_mapping}")
        # =====================================================

        df = pd.concat([x, y], axis=1)
        shared_globals.data = df
        shared_globals.feature_data = x
        shared_globals.feature_names = x.columns.tolist()
        shared_globals.target_name = [df.columns[-1]]
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num = len(shared_globals.data)

        print(f"数据集形状：{x.shape}")
    else:
        print(f"\n原始数据形状：{df.shape}")

        # 2. 优化：保留临床重要特征+筛选特征，避免关键信息丢失
        valid_features = [f for f in shared_globals.feature_data.columns if f in filtered_features]
        # 强制保留临床重要特征（即使不在筛选列表中）
        for feat in clinically_important:
            if feat in shared_globals.feature_data.columns and feat not in valid_features:
                valid_features.append(feat)
                print(f"强制保留临床重要特征: {feat}（不在筛选列表中，但医学意义关键）")

        if not valid_features:
            raise ValueError("筛选的特征在数据中不存在")

        if filtered_features != valid_features:
            print(f"警告：筛选特征顺序已调整为与原始数据一致，最终特征列表：{valid_features}")

        x = shared_globals.feature_data[valid_features]
        y = df.iloc[:, -1]

        # 处理非数值型特征（复用全局编码器）
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"发现非数值型特征，使用预存编码器编码：{non_numeric_cols}")
            for col in non_numeric_cols:
                if col in shared_globals.label_encoders:
                    x[col] = shared_globals.label_encoders[col].transform(x[col])
                else:
                    raise ValueError(f"未找到特征{col}的编码器，需先运行全特征训练")

        # 处理缺失值
        if x.isnull().any().any():
            print(f"检测到缺失值，使用全特征训练时的均值填充...")
            # 复用全特征的均值（避免筛选后均值偏移）
            fill_vals = {col: shared_globals.feature_stats['mean'][col] for col in x.columns}
            x = x.fillna(fill_vals)

        # ========== 修复：筛选特征分支也统一初始化target_encoder ==========
        # 处理目标变量（复用或重新初始化encoder）
        if hasattr(shared_globals, 'target_encoder') and shared_globals.target_encoder is not None:
            # 复用已有的encoder（确保编码一致性）
            y_le = shared_globals.target_encoder
            y_encoded = y_le.transform(y.values)
            print(f"复用已有的target_encoder，对目标变量进行编码")
        else:
            # 重新初始化encoder（首次运行筛选分支时）
            y_le = LabelEncoder()
            y_encoded = y_le.fit_transform(y.values)
            shared_globals.target_encoder = y_le
            print(f"新建target_encoder，对目标变量进行编码")

        # 重构Series
        y = pd.Series(
            y_encoded,
            index=y.index,
            name=y.name
        )
        class_mapping = dict(zip(y_le.classes_, y_le.transform(y_le.classes_)))
        print(f"目标变量编码完成，原始类别→编码值映射: {class_mapping}")
        # =====================================================

        df = pd.concat([x, y], axis=1)
        shared_globals.data = df
        shared_globals.feature_data = x
        shared_globals.feature_names = x.columns.tolist()
        shared_globals.target_name = [df.columns[-1]]
        shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
        shared_globals.sample_num = len(shared_globals.data)

        print(f"筛选后数据集形状：{x.shape}")
        print(f"使用筛选后的特征：{valid_features}")

    # 3. 初始化全局编码器存储（避免后续报错）
    if not hasattr(shared_globals, 'label_encoders'):
        shared_globals.label_encoders = {}
    return x, y


def build_lstm_model(input_shape, num_classes):
    """构建适配cuDNN的LSTM模型（修复加速问题）"""
    inp = Input(shape=input_shape)
    num_features = input_shape[0]

    # 4. 优化：根据特征数量精细调整模型容量（避免过拟合/欠拟合）
    if num_features <= 5:
        units1, units2 = 32, 48  # 特征少→减小单元数，避免过拟合
    elif 5 < num_features <= 10:
        units1, units2 = 48, 64
    else:  # 特征多（如12个）→适度增加单元数，保证拟合能力
        units1, units2 = 64, 80

    # 5. 修复：适配cuDNN的LSTM参数（activation=tanh, 无recurrent_dropout）
    # cuDNN要求：activation=tanh, recurrent_activation=sigmoid, 禁用recurrent_dropout
    x = LSTM(
        units=units1,
        return_sequences=True,
        kernel_initializer='he_normal',
        activation='tanh',  # 必须为tanh（cuDNN强制）
        recurrent_activation='sigmoid',  # 必须为sigmoid（cuDNN强制）
        # 移除recurrent_dropout（cuDNN不支持）
    )(inp)
    x = Dropout(0.3)(x)  # 降低dropout率，保留更多弱特征信息
    x = BatchNormalization()(x)

    # 第二层LSTM（同规则适配cuDNN）
    x = LSTM(
        units=units2,
        kernel_initializer='he_normal',
        activation='tanh',
        recurrent_activation='sigmoid'
    )(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # 分类层：修复之前的bug（Dropout参数错误）
    dense_units = max(48, int(units2 * 0.6))  # 减小全连接层规模，降低过拟合
    q = Dense(dense_units, kernel_initializer='he_normal', activation='tanh')(x)  # 用tanh保持一致性
    q = Dropout(0.2)(q)  # 进一步降低dropout，保留弱特征交互

    # 输出层：降低正则化强度，避免过度约束
    output = Dense(
        num_classes,
        kernel_regularizer=l2(0.003),  # 从0.005降至0.003，减少约束
        activation='softmax'
    )(q)
    model = Model(inputs=inp, outputs=output)

    # 6. 优化学习率策略（更平缓的衰减，避免前期震荡）
    optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0008,  # 从0.001降至0.0008，降低初始步长
            decay_steps=15000,  # 增加衰减步数，减缓学习率下降
            decay_rate=0.95  # 从0.9升至0.95，更平缓的衰减
        )
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印cuDNN适配提示
    print(f"\nLSTM模型结构（适配cuDNN）：")
    print(f"  输入形状: {input_shape}, LSTM单元数: ({units1}, {units2}), 全连接单元数: {dense_units}")
    return model


def train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=180,  # 增加训练轮次，给足够收敛时间
        batch_size=128,
        early_stopping_monitor='val_accuracy',
        early_stopping_patience=8,
        early_stopping_min_delta=0.0003,  # 进一步降低最小改善阈值，捕捉微小提升
        early_stopping_restore_best=True
):
    """优化训练策略（动态调整早停耐心+学习率衰减）"""
    # 7. 优化：根据特征数量调整早停耐心（特征多→需要更多轮次收敛）
    num_features = x_train.shape[1]
    if num_features > 10:
        early_stopping_patience = 12  # 特征多（如12个）→耐心增至12
        print(f"检测到特征数量较多（{num_features}个），早停耐心调整为{early_stopping_patience}")
    elif num_features <=5:
        early_stopping_patience = 10  # 特征少→耐心适中

    # 早停策略
    early_stopping = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=early_stopping_restore_best,
        verbose=1
    )

    # 8. 优化学习率衰减：更平缓的调整（因子从0.7升至0.8）
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=5,  # 延长衰减触发周期，避免频繁降学习率
        factor=0.8,  # 从0.7升至0.8，减缓学习率下降
        min_lr=1e-7,  # 降低最小学习率，允许后期微调
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
        verbose=1
    )
    end_time = time.perf_counter()

    # 记录训练关键指标到全局变量（用于特征贡献计算验证）
    shared_globals.train_history = {
        'train_acc': history.history['accuracy'][-1],
        'val_acc': history.history['val_accuracy'][-1],
        'best_val_acc': max(history.history['val_accuracy']),
        'best_epoch': np.argmax(history.history['val_accuracy']) + 1
    }
    print(f"\n训练关键指标：最佳验证准确率={shared_globals.train_history['best_val_acc']:.4f}（第{shared_globals.train_history['best_epoch']}轮）")

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
    """评估模型性能（新增特征贡献计算的基础数据输出）"""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")

    # 预测并保存概率分布（用于后续JS散度验证）
    y_pred_probs = model.predict(x_test)
    shared_globals.test_pred_probs = y_pred_probs  # 保存预测概率到全局变量
    shared_globals.test_true_labels = np.argmax(y_test, axis=1)

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

    # ========== 修复：安全计算类别级准确率 ==========
    # 计算类别级准确率（辅助分析弱特征对特定类别的影响）
    class_acc = {}
    # 安全获取原始类别（处理encoder未初始化的极端情况）
    if hasattr(shared_globals, 'target_encoder') and shared_globals.target_encoder is not None:
        original_classes = shared_globals.target_encoder.classes_
        print(f"\n类别级准确率（原始类别基于target_encoder解码）：")
    else:
        original_classes = np.unique(y_test_true)
        print(f"\n类别级准确率（target_encoder未初始化，使用预测类别作为原始类别）：")

    # 遍历所有唯一类别，计算准确率
    for cls in np.unique(y_test_true):
        cls_mask = (y_test_true == cls)
        class_acc[cls] = np.mean(y_pred[cls_mask] == cls)
        # 安全获取原始类别名称/值
        if len(original_classes) > cls:  # 避免索引越界
            cls_original = original_classes[cls]
        else:
            cls_original = cls
        # 格式化输出（区分数值型和非数值型原始类别）
        print(f"  编码类别{cls} → 原始值{cls_original}: 准确率={class_acc[cls]:.4f}")
    # =====================================================

    if shared_globals.filtered_features is not None:
        tmp = shared_globals.accuracy
        shared_globals.accuracy = report_dict['accuracy']
        shared_globals.accuracy = 100 * (shared_globals.accuracy - tmp)
        print(f"特征选择后准确率变化: {shared_globals.accuracy:.2f}%")
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

    # 绘制训练历史（标注最佳epoch）
    plt.figure(figsize=(12, 5))
    best_epoch = np.argmax(history.history['val_accuracy']) + 1

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'最佳epoch={best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label=f'最佳epoch={best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return test_acc


def save_model(model, data_path, model_name):
    """保存模型（新增模型结构和参数记录）"""
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    save_filename = f"{dataset_name}_{model_name}.keras"
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, save_filename)

    try:
        model.save(model_path)
        print(f"\nLSTM模型已成功保存至: {os.path.abspath(model_path)}")
        # 保存模型配置到全局变量（用于后续特征扰动时的模型参数参考）
        shared_globals.model_config = {
            'input_shape': model.input_shape[1:],
            'num_classes': model.output_shape[1],
            'lstm_units': [layer.units for layer in model.layers if isinstance(layer, LSTM)],
            'dense_units': [layer.units for layer in model.layers if isinstance(layer, Dense) and layer.name != 'dense_2']
        }
        shared_globals.model = model
        return True
    except Exception as e:
        print(f"\nLSTM模型保存失败: {str(e)}")
        return False


def train(data_path, filtered_features):
    import shared_globals
    # 初始化全局变量（避免特征扰动时变量未定义）
    if not hasattr(shared_globals, 'label_encoders'):
        shared_globals.label_encoders = {}
    if not hasattr(shared_globals, 'target_encoder'):
        shared_globals.target_encoder = None
    if not hasattr(shared_globals, 'feature_stats'):
        shared_globals.feature_stats = {}
    shared_globals.filtered_features = filtered_features  # 记录当前筛选特征

    set_seed(shared_globals.random_seed)

    x, y = load_and_preprocess_data(data_path, filtered_features)

    # 分割数据集
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(
        x, y, test_size=0.2, random_state=shared_globals.random_seed, stratify=y  # 新增stratify，保证类别分布一致
    )
    print(f"\n数据集分割后：")
    print(f"  训练集（原始）: {x_train_raw.shape}, 类别分布: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"  测试集: {x_test.shape}, 类别分布: {pd.Series(y_test).value_counts().to_dict()}")

    # 对训练集进行过采样（仅对训练集！）
    sm = SVMSMOTE(random_state=shared_globals.random_seed, k_neighbors=5)  # 调整k_neighbors，避免过采样噪声
    x_train, y_train = sm.fit_resample(x_train_raw, y_train_raw)
    print(f"过采样后训练集: {x_train.shape}, 类别分布: {pd.Series(y_train).value_counts().to_dict()}")

    # 标准化特征（保存scaler到全局变量，用于后续特征扰动）
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    shared_globals.scaler = scaler  # 保存标准化器，确保扰动时尺度一致

    # 调整形状
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print(f"LSTM输入形状: 训练集={x_train.shape}, 测试集={x_test.shape}")

    # 标签进行独热编码
    onehot = OneHotEncoder(sparse_output=False, categories='auto')
    # y_train = onehot.fit_transform(y_train.reshape(-1, 1))
    # y_test = onehot.transform(y_test.reshape(-1, 1))
    # 1. y_train：Series → NumPy数组 → reshape为二维（n_samples, 1）
    y_train_np = y_train.values.reshape(-1, 1)  # 关键：先转数组再reshape
    y_train = onehot.fit_transform(y_train_np)
    # 2. y_test：同样处理，避免后续报错
    y_test_np = y_test.values.reshape(-1, 1)  # 同步修复y_test
    y_test = onehot.transform(y_test_np)
    shared_globals.onehot_encoder = onehot  # 保存独热编码器

    # 构建模型
    num_classes = y_train.shape[1]
    model = build_lstm_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    # model.summary()  # 可选：打印模型结构

    # 训练模型（根据特征数量动态调整参数）
    print("\n开始训练LSTM模型...")
    if x_train.shape[1] <= 5:
        model, history = train_model(model, x_train, y_train, x_test, y_test,
                                     epochs=220, early_stopping_patience=12)
    elif 5 < x_train.shape[1] <= 10:
        model, history = train_model(model, x_train, y_train, x_test, y_test,
                                     epochs=200, early_stopping_patience=10)
    else:
        model, history = train_model(model, x_train, y_train, x_test, y_test,
                                     epochs=180, early_stopping_patience=12)

    # 评估模型
    print("\n评估LSTM模型性能...")
    test_acc = evaluate_model(model, x_test, y_test, history)

    # 保存模型
    save_model(model, data_path, model_name="lstm_model")

    return model