# shared_globals.py

# 声明全局变量
model = None  # 模型实例
data = None  # 原始数据集（含目标列）
feature_data = None  # 特征数据（不含目标列）
feature_names = []  # 特征名称列表
target_name=[]
all_names = []  # 所有节点名称（含target）

feature_type = {}  # 特征类型字典
feature_values = {}  # 特征取值集合
baselines = None  # 基线值（DataFrame）
baseline_preds = None  # 基线预测值（ndarray）
con_list = []  # 特征贡献值
feature_max_abs = []  # 最大绝对差异

valid_paths = []  # 有效路径

direct_paths = []  # 直接路径（字符串列表）
indirect_paths = []  # 间接路径（字符串列表）

x_list = []  # 扰动值集合
y_list = []  # 预测差异集合
class_weights=None
num_classes=0

path_separators = ['→', '->', '→', '=>']
tolerance = 1e-6  # 概率容忍误差
random_seed = 42  # 随机种子
epsilon = 1e-10  # 避免除零
original_samples = None  # 原始样本（DataFrame）
original_preds = None  # 原始样本预测值（ndarray）
subset_results = []  # 中间结果列表
feature_effects = {}  # 存储特征的直接效应和间接效应汇总结果
feature_noise={}
filtered_features=None
redundant_features=[]
redundant_set=[]
corr_matrix=None
sample_num=0

accuracy = 0
f_score = 0
running_time = 0
features=[]
critical_interaction_sets=[]
perturbation_instance=None

causal_graph_builder=None
high_contrib_features=[]
all_target_categories=None
significant_feature_pairs_log = []
to_add = []

originalreport={}
afterreport={}
tnr_font=None


def init():
    """初始化全局变量"""
    global model,data, feature_data,  feature_names, target_name ,feature_type
    global feature_values, baselines, baseline_preds, con_list, feature_max_abs
    global valid_paths, all_names, direct_paths, indirect_paths, x_list, y_list
    global path_separators, tolerance, random_seed, epsilon, original_samples, original_preds
    global subset_results
    global feature_effects,feature_noise,filtered_features
    global accuracy,f_score,running_time
    global redundant_features,redundant_set,corr_matrix,sample_num,features,critical_interaction_sets,causal_graph_builder
    global perturbation_instance,high_contrib_features
    global class_weights,num_classes,all_target_categories
    global significant_feature_pairs_log,to_add,originalreport,afterreport, tnr_font

    model = None
    feature_data = None
    feature_names = []
    target_name = []
    feature_type = {}
    feature_values = {}
    original_samples = None
    original_preds = None  # 初始化为None，而非列表

    baselines = None
    baseline_preds = None

    con_list = []
    x_list = []
    y_list = []
    feature_max_abs = []
    valid_paths = []
    all_names = []
    direct_paths = []
    indirect_paths = []

    path_separators = ['→', '->', '→', '=>']
    tolerance = 1e-6
    random_seed = 42
    epsilon = 1e-10
    subset_results = []
    feature_effects = {}  # 存储特征的直接效应和间接效应汇总结果
    feature_noise={}

    filtered_features=None
    redundant_features=[]
    redundant_set=[]
    corr_matrix=None
    sample_num=0

    features=[]
    critical_interaction_sets=[]
    causal_graph_builder=None
    perturbation_instance=None
    high_contrib_features=[]

    accuracy = 0
    f_score = 0
    running_time = 0

    class_weights=None
    num_classes=0

    all_target_categories=None

    significant_feature_pairs_log = []
    to_add=[]
    originalreport={}
    afterreport={}
    tnr_font=None