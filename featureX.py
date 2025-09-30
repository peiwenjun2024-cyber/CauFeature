import random
import time
import copy
import scipy.stats as stats
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，仅保存图形不显示
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from scipy.stats import spearmanr
import pandas as pd

data_path="dataset/BrainMethod.csv"
model_save_path="featureX_model\BrainMethod.keras"
targetname="is_brain"
df = pd.read_csv(data_path, encoding="gbk")
x, y = df.iloc[:, 0:-1], df.iloc[:, -1]

from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)

# columns_to_drop = F_remove
# df_dropped = x.drop(columns=columns_to_drop)
# x = df_dropped


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))
onehot = OneHotEncoder(sparse_output=False)
y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))


inp=Input(shape=(x_train.shape[1:]))

print(inp)

x = Conv1D(32, 3, padding = "same", activation='tanh')(inp)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=3, strides=1, padding='same',activation='relu')(x)#padding可能会导致靠近边界的部分相比于中间部分对于输出的的影响更小
                                                                              #可能会导致边界存在一定程度的欠表示
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=3, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=3, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
g = Flatten()(x)
output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='softmax')(g)

model = Model(inputs=inp,outputs=output)
print(output)

model.compile(optimizer = tf.keras.optimizers.Adam(0.001),  #优化器
              loss = 'categorical_crossentropy', #损失函数
              metrics = ['accuracy']
             )

lr_reduce=keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.000001)

start = time.perf_counter()
history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    callbacks = [lr_reduce],
                    batch_size= 128,
                    validation_data = (x_test,y_test),
                    shuffle=True)
end = time.perf_counter()




print(model.evaluate(x_test,y_test))
print('Running time: %s Seconds'%(end-start))
Y_test = np.argmax(y_test, axis=1)# Convert one-hot to index
from sklearn.metrics import classification_report
predict = model.predict(x_test)
y_pred=np.argmax(predict,axis=1)
print(classification_report(Y_test, y_pred,digits=5))


model.save(model_save_path)

def set_feature_dict(column_names,feature_dict): # Assigning quantitative information
    '''
    column_names (list): 包含特征名称的列表。
    feature_dict (dict): 用于存储每个特征量化信息的字典。
    - 'change_range' ：初始值设为 0，可能用于表示特征的变化范围。
    - 'correlation' ：初始值设为 False ，可能用于标记该特征是否存在相关性。
    - 'target' ：初始值设为 0，可能用于表示该特征与目标变量的某种关联值。
    '''
    for feature in column_names:
        feature_dict[feature] = {'change_range':0,'correlation': False,'target':0}
    return feature_dict

def Feature_correlation_analysis(column_names,feature_dict,spearmanr_data): # Correlation between features
    Feature_correlation_list = []
    correlation_matrix, _ = spearmanr(spearmanr_data, axis=0)
    threshold = 0.7
    correlation_matrix[np.abs(correlation_matrix) < threshold] = np.nan #设定阈值 0.7 ，将绝对值小于该阈值的相关系数置为 nan ，以此过滤掉弱相关性。
    plt.figure(figsize=(10, 8)) 
    mask = np.tri(*correlation_matrix.shape, k=-1) #np.tri(*correlation_matrix.shape, k=-1) 生成一个下三角矩阵作为掩码，用于隐藏相关系数矩阵的下三角部分，避免重复显示。
    correlation_matrix = np.ma.array(correlation_matrix, mask=mask) #将相关系数矩阵转换为掩码数组，应用掩码。
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True) #绘制热力图，展示强相关性的特征。
    plt.xticks(range(len(column_names)), column_names, rotation=45, horizontalalignment='right')
    plt.yticks(range(len(column_names)), column_names)
    plt.title("Spearman")
    rela_class = []
    count = -1
    rows, cols = np.where(~np.isnan(correlation_matrix))
    #这部分代码遍历相关系数矩阵中非 nan 和未被掩码的元素，将具有相关性的特征分组到 rela_class 中。通过 flag 标志判断是否需要创建新的分组。
    for row, col in zip(rows, cols):
        if row != col and correlation_matrix[row, col] != 'masked':
            Feature_correlation_list.append(str(f"Feature {column_names[row]} and Feature {column_names[col]} have correlation: {correlation_matrix[row, col]:.2f}"))
            feature1 = column_names[row]
            feature2 = column_names[col]
            if not rela_class:
                rela_class.append([feature1,feature2])
                continue
            for i in rela_class:
                flag = 0
                count += 1
                if feature1 in i and feature2 not in i :
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
            flag = 0
            count = -1
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
    for i in rela_class:
        for j in i:
            if j in feature_dict:
                feature_dict[j]['correlation'] = True
    F = column_names
    F_copy = copy.deepcopy(F)
    for i in F:
        if feature_dict[i]['correlation'] is False:
            feature_dict[i]['NOFR'] = 0
        else:
            for j in rela_class:
                for metric in j:
                    if metric == i:
                         feature_dict[i]['NOFR'] = len(j)-1    
    return rela_class,feature_dict,Feature_correlation_list

def target_correlation_analysis(feature_dict,target_variable,column_names): # Correlation between features and labels
    #- feature_dict ：用于存储每个特征相关性信息的字典。- target_variable ：目标变量，通常是标签数据。- column_names ：特征名称的列表。注释说明该函数的作用是计算特征与标签之间的相关性。
    target_variable_list = []  #用于存储每个特征与目标变量相关性的描述信息。
    unique_values, counts = np.unique(target_variable.values, return_counts=True)
    if unique_values.shape !=(2,):
        '''
        当目标变量的唯一值数量不为 2 时，使用 spearmanr 函数计算每个特征与目标变量之间的斯皮尔曼等级相关系数。将计算结果存储在 feature_dict 中，并将相关性信息添加到 target_variable_list 里。
        '''
        for feature in column_names:  
            point_biserial_corr, p_value = spearmanr(df[feature], target_variable)
            feature_dict[feature]['target'] = point_biserial_corr
            target_variable_list.append(str(f"{feature} - target_variable: {point_biserial_corr:.2f}"))
    else:
        '''
        当目标变量的唯一值数量为 2 时，使用 stats.pointbiserialr 函数计算每个特征与目标变量之间的点二列相关系数。同样将结果存储在 feature_dict 中，并将相关性信息添加到 target_variable_list 里。
        '''
        for feature in column_names: 
            point_biserial_corr, p_value = stats.pointbiserialr(df[feature], target_variable)
            feature_dict[feature]['target'] = point_biserial_corr
            target_variable_list.append(str(f"{feature} - target_variable: {point_biserial_corr:.2f}"))
    return feature_dict,target_variable_list

def get_col_ranges(data,col_name): # Analysing the values of features   分析特征值
    col_data = data[col_name]    #通过列名获取该列数据
    min_value = col_data.min()
    max_value = col_data.max()
    values_count = col_data.nunique()   #非重复样本数M
    values_range = col_data.unique()    #非重复值Value
    values_range.sort()                 #非重复值升序排列
    values_range = values_range.tolist()   #列表
    return min_value,max_value,values_count,values_range  #返回最小值  最大值 非重复样本值M  非重复样本值Value列表

def value_range(column_names,df): # Getting the feature statistics   获取特征数据
    """ 获取特征统计信息，包括最小值、最大值、值计数和取值范围 """
    min_value_list,max_value_list,values_count_list,values_range_list = [],[],[],[]
    for i in column_names:
        min_value,max_value,values_count,values_range = get_col_ranges(df,i)   #逐列分析
        min_value_list.append(min_value)
        max_value_list.append(max_value)
        values_count_list.append(values_count)
        values_range_list.append(values_range)
    return min_value_list,max_value_list,values_count_list,values_range_list  #获取最小值列表、最大值列表、非重复值样本数列表、非重复样本值Value列表

def Disturbance_range(column_names, min_value_list, max_value_list, values_count_list, values_range_list): # Feature Perturbation   生成特征扰动值的范围

    x_list = []
    for j in range(0,len(column_names)):
        if min_value_list[j]==0 and max_value_list[j]==1 and values_count_list[j]!=2:       # 情况3：处理最小值为 0，最大值为 1，且特征值唯一值数量不为 2 的情况，意指最小值和最大值接近且多样性强
            x_list.append ([i / 10 for i in range(11)])                                     # 设定特征值范围为 0 到 1，步长为 0.1
        elif (max_value_list[j]-min_value_list[j])/values_count_list[j] > 100:      # 情况2：处理 (最大值 - 最小值) / 非重复特征值数量 大于 100
            num_of_random_values = 30                                               # 设定随机生成的特征值数量为 30      
            for _ in range(num_of_random_values):
                random_data = random.randint(min(values_range_list[j]), max(values_range_list[j]))        #生成一个指定范围内的随机整数
                values_range_list[j].append(random_data)
                values_range_list[j].sort()
            x_list.append(values_range_list[j])
        else:
            x_list.append ([i for i in range(int(min_value_list[j]),int(max_value_list[j])+1)])  #其他情况，从最小值到最大值，逐次加一
    return x_list

#from shap.explainers.tf_utils import tf
'''该函数负责计算特征的重要性。它根据不同的条件生成特征值，并通过模型预测计算相应的输出变化。
使用 numpy 和 tensorflow 处理数据，并记录运行时间。'''

def Fresh_breeze(x,column_names,min_value_list,max_value_list,values_count_list,init_val): # Feature importance analysis
    y_list = [[] for _ in range(len(column_names))]   # 为每个特征初始化空列表
    interval = [i / 10 for i in range(11)]            # 生成 [0.0, 0.1, ..., 1.0]
    rel_out = np.argmax(init_val,axis=1)                # 获取初始预测的类别（如二分类的0或1）
    start = time.perf_counter()
    for j in range(len(column_names)):    # 根据特征类型生成候选值 values
        if min_value_list[j] == 0 and max_value_list[j] == 1 and values_count_list[j]!=2:
            values = [interval[i] for i in range(11)]
        elif (max_value_list[j]-min_value_list[j])/values_count_list[j] > 100:
            values = [x_list[j][i] for i in range(len(x_list[j]))]
        elif len(x_list[j]) > 10000:
            num_partitions = 1000
            step = (max_value_list[j]-min_value_list[j])/num_partitions
            values = [min_value_list[j] + k * step for k in range(num_partitions)]
            x_list[j] = values
        else:
            values = [i for i in range(int(min_value_list[j]), int(max_value_list[j]) + 1)]
        for i in values:
            modified_x = list(x.numpy())    # 将张量转为可修改的Python列表
            modified_x[0][j] = i            # 修改特征j的值
            modified_x = tf.convert_to_tensor(modified_x, dtype=tf.float32) # 转回张量
            rel = model.predict(modified_x, verbose=0)             # 获取新预测结果
            variety = rel - init_val                    # 计算变化量
            first_value = variety[0][rel_out]           # 提取目标类别的变化值
            y_list[j].append(first_value)               # 记录结果
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    return y_list

'''用于可视化特征的重要性。它绘制每个特征的预测变化图，并标注最大值和最小值。
使用 matplotlib 进行图形绘制。'''
def visualization(column_names,y_list,x_list):    # column_names为特征名称，y_list为预测变化，x_list为特征值
    y_list_max = [max(sub_list) for sub_list in y_list]      # y_list_max为预测变化的最大值，y_list_min为预测变化的最小值
    y_list_min = [min(sub_list) for sub_list in y_list] 
    y_list_abs = [[abs(x) for x in inner_list] for inner_list in y_list]        # 计算 y_list 中每个元素的绝对值，得到 y_list_abs                                                                                      
    max_values = [max(sublist) for sublist in y_list_abs]
    for i in range(len(column_names)):      # 遍历特征名称列表，对每个特征绘制预测变化图。
        plt.figure(figsize=(4,30))          # 创建一个新的图形，设置图形大小为宽 10、高 80。 
        plt.subplot(len(column_names), 1, i+1)      # 将图形分割为 len(column_names) 行、1 列，当前子图为第 i+1 个子图。
        plt.plot(x_list[i], y_list[i],linestyle='-', color='r')  # 绘制 x_list[i] 和 y_list[i] 的折线图，使用红色实线表示。 
        max_index = y_list[i].index(max(y_list[i]))
        min_index = y_list[i].index(min(y_list[i]))
        x_max, y_max = x_list[i][max_index], y_list[i][max_index]
        x_min, y_min = x_list[i][min_index], y_list[i][min_index]
        #使用 plt.annotate 在图上标注最大值和最小值的位置
        plt.annotate(f'Max: ({x_max}, {y_max})', (x_max, y_max), textcoords="offset points", xytext=(0,10), ha='center', color='red') 
        plt.annotate(f'Min: ({x_min}, {y_min})', (x_min, y_min), textcoords="offset points", xytext=(0,10), ha='center', color='green')
        plt.xlabel(column_names[i])           #设置 x 轴标签为当前特征的名称                                                                                          
        plt.ylabel('Forecast change')           #设置 y 轴标签为 “Forecast change”
        plt.ylim(y_list_min[i]*1.1,y_list_max[i]*1.5)  #设置 y 轴的范围     
        plt.grid(True)                          #显示网格线
        # 关键：绘制完成后关闭当前图形，释放内存
        plt.close()  # 添加这一行
    return max_values                         #返回 y_list_abs 中每个子列表的最大值列表，返回每个特征预测变化绝对值的最大值列表

'''该函数分析特征的重要性，并根据给定的阈值合并较小的重要性值，最后生成条形图和饼图进行可视化。'''
def feature_important(column_names,threshold,max_values): # Visual feature importance analysis 
    #- column_names ：特征名称列表。- threshold ：合并较小重要性值的阈值（百分比）。- max_values ：每个特征的最大重要性值列表
    total_sum = sum(max_values)   # 计算特征最大预测变化总和
    percentages = [(value / total_sum) * 100 for value in max_values]    #通过列表推导式计算每个特征重要性值占总和的百分比
    merged_values = []                                                      # 初始化合并后的重要性值列表
    merged_categories = []                                                  # 初始化合并后的特征名称列表
    other_total = 0         # 初始化"others"类别的总重要性值
    # 遍历特征名称和最大重要性值列表，将小于阈值的重要性值合并到"others"类别中
    for category, value in zip(column_names, max_values):
        if value / total_sum * 100 < threshold:  # 检查当前特征的重要性值占总和的百分比是否小于阈值
            other_total += value               # 将当前特征的重要性值累加到"others"类别的总重要性值中
        else:
            merged_values.append(value)
            merged_categories.append(category)

    merged_values.append(other_total)
    merged_categories.append("others")
    merged_values = np.concatenate(merged_values)   #拼接特征
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))      # 创建一个包含两个子图的图形
    axs[0].plot(merged_categories, merged_values, marker='o', color='#813C85', label='line')    # 绘制折线图
    axs[0].bar(merged_categories, merged_values, alpha=0.4, color='#DE3F7C', label='bar')       # 绘制柱状图
    axs[0].tick_params(axis='x', rotation=70)                 # 调整x轴标签的旋转角度                                            
    axs[1].pie(merged_values, labels=merged_categories, autopct='%1.1f%%')              # 绘制饼图
    plt.show()          # 显示图形  


def feature_selection(column_names, feature_dict, rela_class,
                      max_values):  # - 函数接收四个参数：`column_names`（特征名称列表）、`feature_dict`（存储特征信息的字典）、`rela_class`（特征关联类列表）和 `max_values`（特征变化范围值列表）。

    for i in range(len(column_names)):
        feature_dict[column_names[i]]['change_range'] = max_values[
            i]  # - 遍历 `column_names`，将 `max_values` 中的值赋给 `feature_dict` 里对应特征的 `change_range` 字段。
    Rela_class_copy = copy.deepcopy(rela_class)
    feature_dict_copy = copy.deepcopy(feature_dict)  # - 函数首先将 `rela_class` 和 `feature_dict` 进行深拷贝，以避免对原始数据的修改。
    F = column_names
    F_copy = copy.deepcopy(F)  # - `F` 赋值为 `column_names`，并复制一份到 `F_copy`。
    flag = 0
    num = 0  # - 初始化 `flag` 和 `num` 变量，`flag` 用于控制内层循环的跳出，`num` 用于统计低贡献特征的数量。
    for i in range(len(column_names)):  # Judging the number of low—contributing features
        if feature_dict[column_names[i]]['change_range'] >= 0.1:  # 遍历 column_names ，统计 change_range 大于等于 0.1 的特征数量。
            num += 1
    if (num / len(column_names)) >= 0.15:  # case1
        for i in range(len(rela_class)):
            for metric in rela_class[i]:  # Removal of features
                if feature_dict[metric]['change_range'] >= 0.5 or feature_dict[metric]['change_range'] <= 0.1:
                    Rela_class_copy[i].remove(metric)  # - 移除 `Rela_class_copy` 中 `change_range` 大于等于 0.5 或小于等于 0.1 的特征
                    for j in Rela_class_copy[i]:
                        feature_dict_copy[j]['NOFR'] -= 1  # 更新剩余特征的 `NOFR`。
        for metric in F:
            if feature_dict[metric]['change_range'] <= 0.1:  # Removal of low—contributing features
                F_copy.remove(metric)  # - 从 `F_copy` 中移除 `change_range` 小于等于 0.1 的低贡献特征。
            elif 0.1 < feature_dict[metric]['change_range'] < 0.5 and -0.3 < feature_dict[metric]['target'] < 0.3 and \
                    feature_dict_copy[metric]['NOFR'] > 0:
                # Filtering features based on correlation 
                # - 对于满足特定条件（`change_range` 在 0.1 到 0.5 之间，`target` 在 -0.3 到 0.3 之间，且 `NOFR` 大于 0）的特征，根据相关性进行筛选，只保留关联类中 `change_range` 最大的特征。
                for i in Rela_class_copy:
                    for j in i:
                        if metric == j:
                            flag = 1
                            change_range = [feature_dict[feature]['change_range'] for feature in i]
                            max_change_range = max(change_range)
                            max_feature = \
                            [feature for feature in i if feature_dict[feature]['change_range'] == max_change_range][0]
                            for feature in i:
                                feature_dict_copy[feature]['NOFR'] = 0
                                if feature != max_feature:
                                    if feature in F_copy:
                                        F_copy.remove(feature)
                            break
                    if flag == 1:
                        break
                flag = 0
    else:  # case2  #低贡献特征比例小于 0.15**
        for i in range(len(rela_class)):
            for metric in rela_class[i]:
                if feature_dict[metric]['change_range'] >= 0.1:
                    Rela_class_copy[i].remove(metric)  # - 移除 `Rela_class_copy` 中 `change_range` 大于等于 0.1 的特征
                    for j in Rela_class_copy[i]:
                        feature_dict_copy[j]['NOFR'] -= 1  # 更新剩余特征的 `NOFR`
        for metric in F:
            if -0.7 < feature_dict[metric]['target'] < 0.7 and feature_dict_copy[metric]['NOFR'] > 0:
                # - 对于满足特定条件（`target` 在 -0.7 到 0.7 之间，且 `NOFR` 大于 0）的特征，根据相关性进行筛选，只保留关联类中 `change_range` 最大的特征。
                for i in Rela_class_copy:
                    for j in i:
                        if metric == j:
                            flag = 1
                            change_range = [feature_dict[feature]['change_range'] for feature in i]
                            max_change_range = max(change_range)
                            max_feature = \
                            [feature for feature in i if feature_dict[feature]['change_range'] == max_change_range][0]
                            for feature in i:
                                feature_dict_copy[feature]['NOFR'] = 0
                                if feature != max_feature:
                                    if feature in F_copy:
                                        F_copy.remove(feature)
                            break
                    if flag == 1:
                        break
                flag = 0
    F_remove = [item for item in F if item not in F_copy]
    return F_remove  # 通过列表推导式找出 F 中不在 F_copy 里的特征，即需要移除的特征列表，并返回该列表。
    
    
# ---------------------- 核心模块：时间记录与计算（重点修改区域） ----------------------
# 初始化总时间变量
total_time = 0.0
# ---------- 1. 相关性分析模块计时（Feature_correlation_analysis + target_correlation_analysis） ----------


from tensorflow.keras.models import load_model
model = load_model(model_save_path)       #加载CNN模型
df = pd.read_csv(data_path, encoding="gbk")
column_names = df.columns[0:-1].tolist()

print("\n=== 开始相关性分析 ===")
corr_start = time.perf_counter()  # 记录开始时间

feature_dict = {}
spearmanr_data = df.iloc[:,0:-1]

feature_dict = set_feature_dict(column_names,feature_dict)
rela_class,feature_dict,feature_correlation_list = Feature_correlation_analysis(column_names,feature_dict,spearmanr_data)

target_variable = df[targetname]
column_names = df.columns[0:-1]
feature_dict,target_variable_list = target_correlation_analysis(feature_dict,target_variable,column_names)

corr_end = time.perf_counter()  # 记录结束时间
corr_time = corr_end - corr_start
total_time += corr_time
print(f"相关性分析模块耗时: {corr_time:.4f} Seconds")


df = pd.read_csv(data_path, encoding="gbk")
base = df.iloc[:,0:-1]
base_data = base.mean()
base_data = base_data.round().astype(int)   #对平均值进行四舍五入并转为整数
base_data = base_data.values.reshape(1,base_data.shape[0],1)  #将数据重塑为 3D 数组,将 NumPy 数组重塑为形状为 (1, n, 1) 的三维数组，其中 n 是平均值的数量。这里的 1 表示样本数量为 1，n 是特征数量（即列数），而最后一个 1 表示每个特征的维度。
x = base_data

import tensorflow as tf
x = tf.convert_to_tensor(x,dtype=tf.float32)  #x 现在是一个 TensorFlow 张量，适合直接用于模型的输入
init_val = model.predict(x) 

# ---------- 2. 特征扰动模块计时（Disturbance_range） ----------
print("\n=== 开始特征扰动 ===")
disturb_start = time.perf_counter()  # 记录开始时间


min_value_list,max_value_list,values_count_list,values_range_list = value_range(column_names,df)
x_list = Disturbance_range(column_names,min_value_list,max_value_list,values_count_list,values_range_list)

disturb_end = time.perf_counter()  # 记录结束时间
disturb_time = disturb_end - disturb_start
total_time += disturb_time
print(f"特征扰动模块耗时: {disturb_time:.4f} Seconds")

# ---------- 3. 重要性评估模块计时（Fresh_breeze + visualization） ----------
print("\n=== 开始重要性评估 ===")
importance_start = time.perf_counter()  # 记录开始时间

y_list = Fresh_breeze(x,column_names,min_value_list,max_value_list,values_count_list,init_val)

max_values = visualization(column_names,y_list,x_list)

threshold = 2
feature_important(column_names,threshold,max_values)

importance_end = time.perf_counter()  # 记录结束时间
importance_time = importance_end - importance_start
total_time += importance_time
print(f"重要性评估模块耗时: {importance_time:.4f} Seconds")


column_names = df.columns[0:-1].tolist()

# ---------- 4. 特征恢复模块计时（feature_selection） ----------
print("\n=== 开始特征恢复 ===")
selection_start = time.perf_counter()  # 记录开始时间


F_remove = feature_selection(column_names,feature_dict,rela_class,max_values)

selection_end = time.perf_counter()  # 记录结束时间
selection_time = selection_end - selection_start
total_time += selection_time
print(f"特征恢复模块耗时: {selection_time:.4f} Seconds")


df = pd.read_csv(data_path, encoding="gbk")
x, y = df.iloc[:, 0:-1], df.iloc[:, -1]

from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE(random_state=42)
x, y = sm.fit_resample(x, y)

columns_to_drop = F_remove
df_dropped = x.drop(columns=columns_to_drop)
x = df_dropped


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1],1))
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = onehot.fit_transform(y_test.values.reshape(len(y_test), 1))

inp=Input(shape=(x_train.shape[1:]))

print(inp)

x = Conv1D(32, 3, padding = "same", activation='tanh')(inp)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=3, strides=1, padding='same',activation='relu')(x)

x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=3, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=3, strides=1, padding='same',activation='relu')(x)
x = LeakyReLU(alpha=0.33)(x)
x = Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
g = Flatten()(x)
output = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='softmax')(g)

model = Model(inputs=inp,outputs=output)

model.compile(optimizer = tf.keras.optimizers.Adam(0.001),  #优化器
              loss = 'categorical_crossentropy', #损失函数
              metrics = ['accuracy']
             )

lr_reduce=keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.000001)

start = time.perf_counter()
history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    callbacks = [lr_reduce],
                    batch_size= 128,
                    validation_data = (x_test,y_test),
                    shuffle=True)
end = time.perf_counter()

print(model.evaluate(x_test,y_test))
print('Running time: %s Seconds'%(end-start))
Y_test = np.argmax(y_test, axis=1)# Convert one-hot to index
from sklearn.metrics import classification_report
predict = model.predict(x_test)
y_pred=np.argmax(predict,axis=1)
print(classification_report(Y_test, y_pred,digits=5))


# ---------- 总计时间汇总 ----------
print("\n" + "="*50)
print("四大模块耗时汇总:")
print(f"1. 相关性分析: {corr_time:.4f} Seconds")
print(f"2. 特征扰动: {disturb_time:.4f} Seconds")
print(f"3. 重要性评估: {importance_time:.4f} Seconds")
print(f"4. 特征恢复: {selection_time:.4f} Seconds")
print(f"{'='*50}")
print(f"四大模块总计耗时: {total_time:.4f} Seconds")
print("="*50)


