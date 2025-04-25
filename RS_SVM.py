import pandas as pd
import numpy as np
import os
import time
import threading
import joblib
import psutil
import logging
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

# ==================== 全局配置 ====================
DATA_DIR = r"C:\Users\Jackson\Desktop\毕设"
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "optimized_RS_RBF_SVM_model.pkl")
SAFE_MEMORY_LIMIT = 10 * 1024**3

# 路径配置
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RS_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== 数据集特征配置 ====================
UNSW_FEATURES = {
    'id': 'int32', 'dur': 'float32', 'proto': 'category', 'service': 'category',
    'state': 'category', 'spkts': 'int16', 'dpkts': 'int16', 'sbytes': 'int32',
    'dbytes': 'int32', 'rate': 'float32', 'sttl': 'int16', 'dttl': 'int16',
    'sload': 'float32', 'dload': 'float32', 'sloss': 'int16', 'dloss': 'int16',
    'sinpkt': 'float32', 'dinpkt': 'float32', 'sjit': 'float32', 'djit': 'float32',
    'swin': 'int32', 'stcpb': 'int32', 'dtcpb': 'int32', 'dwin': 'int32',
    'tcprtt': 'float32', 'synack': 'float32', 'ackdat': 'float32', 'smean': 'float32',
    'dmean': 'float32', 'trans_depth': 'int16', 'response_body_len': 'int32',
    'ct_srv_src': 'int16', 'ct_state_ttl': 'int16', 'ct_dst_ltm': 'int16',
    'ct_src_dport_ltm': 'int16', 'ct_dst_sport_ltm': 'int16', 'ct_dst_src_ltm': 'int16',
    'is_ftp_login': 'int8', 'ct_ftp_cmd': 'int16', 'ct_flw_http_mthd': 'int16',
    'ct_src_ltm': 'int16', 'ct_srv_dst': 'int16', 'is_sm_ips_ports': 'int8',
    'attack_cat': 'category', 'label': 'int8'
}

# ==================== 随机搜索配置 ====================
RS_CONFIG = {
    'n_iter': 30                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ,
    'cv': 3,
    'scoring': 'f1_weighted',
    'n_jobs': max(psutil.cpu_count()//2, 1),
    'verbose': 4,
    'random_state': 42
}

SVM_PARAM_DIST = {
    'C': loguniform(0.1, 10),
    'gamma': loguniform(1e-4, 1e+1),
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovo', 'ovr'],
    'shrinking': [True, False]
}

# ==================== 内存监控 ====================
class AdvancedMemoryGuard:
    def __init__(self):
        self.process = psutil.Process()
        self.warn_threshold = SAFE_MEMORY_LIMIT * 0.8
        self.peak_sys_cpu = 0
        self.peak_proc_cpu = 0
        self.peak_mem = 0
        self.sys_cpu_samples = []
        self.proc_cpu_samples = []
        self._monitor_active = True
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self):
        """后台监控循环"""
        while self._monitor_active:
            self.check()
            time.sleep(1)

    def stop(self):
        """停止监控线程"""
        self._monitor_active = False

    def check(self):
        """检查资源使用（线程安全）"""
        with self.lock:
            # ===== 内存检查 =====
            current_mem = self.process.memory_info().rss
            # 累加所有子进程的内存
            for child in self.process.children(recursive=True):
                try:
                    current_mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            self.peak_mem = max(self.peak_mem, current_mem)
            
            if current_mem > SAFE_MEMORY_LIMIT:
                raise MemoryError(f"内存超限! 当前: {current_mem/1024**3:.2f}GB / 限制: {SAFE_MEMORY_LIMIT/1024**3:.2f}GB")
            elif current_mem > self.warn_threshold:
                logging.warning(f"内存使用超过80%: {current_mem/1024**3:.2f}GB")

            # ===== CPU检查 =====
            # 系统级CPU（阻塞式测量，确保准确性）
            sys_cpu = psutil.cpu_percent(interval=0.5)
            # 进程级CPU（主进程+所有子进程）
            proc_cpu = self.process.cpu_percent()
            for child in self.process.children(recursive=True):
                try:
                    proc_cpu += child.cpu_percent()
                except psutil.NoSuchProcess:
                    continue

            # 更新数据
            self.sys_cpu_samples.append(sys_cpu)
            self.proc_cpu_samples.append(proc_cpu)
            self.peak_sys_cpu = max(self.peak_sys_cpu, sys_cpu)
            self.peak_proc_cpu = max(self.peak_proc_cpu, proc_cpu)

            logging.info(
                "资源监控:\n"
                f"  System CPU : Current={sys_cpu:5.1f}% | Peak={self.peak_sys_cpu:5.1f}%\n"
                f"  Process CPU: Current={proc_cpu:5.1f}% | Peak={self.peak_proc_cpu:5.1f}%\n"
                f"  Memory Usage: {current_mem/1024**3:.2f}GB / {SAFE_MEMORY_LIMIT/1024**3:.2f}GB"
            )
            return current_mem

    def get_cpu_stats(self):
        """获取CPU统计信息（线程安全）"""
        with self.lock:
            return {
                'sys_avg': np.mean(self.sys_cpu_samples) if self.sys_cpu_samples else 0,
                'sys_peak': self.peak_sys_cpu,
                'proc_avg': np.mean(self.proc_cpu_samples) if self.proc_cpu_samples else 0,
                'proc_peak': self.peak_proc_cpu,
                'peak_mem': self.peak_mem
            }

# ==================== 数据预处理器 ====================
class EnhancedPreprocessor:
    
    def __init__(self):
        self.cat_mapping = {}
        self.num_stats = {}
        self.scaler = StandardScaler()  
        self.feature_columns = []  
        self.label_categories_ = None 
        self.feature_selector = None
        self.column_transformer = None
        self.label_categories_ = None
        self.numeric_features = []
        self.categorical_features = ['proto', 'service', 'state']
        
        # 配置日志记录器
        self.logger = logging.getLogger('Preprocessor')
        self.logger.setLevel(logging.INFO)    
        
    def _get_feature_names(self, transformer, input_features):
        """确保特征名称传递"""
        if hasattr(transformer, 'get_feature_names'):
            return transformer.get_feature_names(input_features)
        return input_features
        
    def _engineer_attack_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """安全生成新特征（防止除零错误）"""
        eps = 1e-6
        
        # 流量强度特征
        X['total_packets'] = X['spkts'] + X['dpkts']
        X['total_bytes'] = X['sbytes'] + X['dbytes']
        
        # 时序特征（添加极小值防止除零）
        X['resp_time_ratio'] = np.where(
            (X['ackdat'] == 0) | (X['synack'] == 0),
            0,
            (X['synack'] + eps) / (X['ackdat'] + eps)
        )
        
        # 协议交互特征
        X['tcp_udp_interaction'] = X['ct_srv_src'] * X['ct_srv_dst']
        
        # 数据包异常比率
        X['abnormal_pkt_ratio'] = (X['sbytes'] + eps) / (X['dbytes'] + eps)
        
        # 会话持续时间特征
        X['flow_duration_per_packet'] = np.where(
            X['total_packets'] == 0,
            0,
            X['dur'] / (X['total_packets'] + eps)
        )
        
        return X

    def _advanced_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        # 检查必需字段是否存在
        required = ['ct_src_ltm', 'ct_dst_ltm', 'spkts', 'dpkts']
        missing = [col for col in required if col not in X.columns]
        if missing:
            raise ValueError(f"缺失必要字段: {missing}")
        
        # 保留原始数据副本
        X_eng = X.copy()
        
       # 1. 避免除零错误
        X_eng['total_packets'] = X_eng['spkts'] + X_eng['dpkts']
        X_eng['total_bytes'] = X_eng['sbytes'] + X_eng['dbytes']
        X_eng['resp_time_ratio'] = np.where(
            (X_eng['ackdat'] == 0) | (X_eng['synack'] == 0),
            0,
            (X_eng['synack'] + 1e-6) / (X_eng['ackdat'] + 1e-6)
        )
        
        # 2. 增加带随机噪声的特征
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(X_eng))
        X_eng['tcp_udp_interaction'] = X_eng['ct_srv_src'] * X_eng['ct_srv_dst'] + noise
        X_eng['abnormal_pkt_ratio'] = (X_eng['sbytes'] + 1) / (X_eng['dbytes'] + 1) + noise
        
        # 3. 数据包分布特征
        X_eng['src_dst_pkt_ratio'] = self._handle_zero_division(
            X_eng['spkts'] + 1, 
            X_eng['dpkts'] + 1
        )
        
        # 4. 端口交互特征
        X_eng['port_entropy'] = (X_eng['ct_src_ltm'] * X_eng['ct_dst_ltm']) ** 0.5
        
        # 5. 协议特定统计
        proto_stats = X_eng.groupby('proto', observed=False)[['sbytes', 'dbytes']].transform('mean')
        X_eng['proto_sbytes_ratio'] = X_eng['sbytes'] / (proto_stats['sbytes'] + 1)
        X_eng['proto_dbytes_ratio'] = X_eng['dbytes'] / (proto_stats['dbytes'] + 1)
        
        self.logger.info(f"生成新特征: {list(X_eng.columns[len(X.columns):])}")
        return X_eng
    
    def _build_preprocessing_pipeline(self):
        """构建优化后的预处理管道"""
        return ColumnTransformer([
            ('num_pipeline', Pipeline([
                ('variance_threshold', VarianceThreshold(threshold=0.0)),  # 允许零方差
                ('scaler', StandardScaler()),
                ('power_transform', PowerTransformer(method='yeo-johnson'))
            ]), self.numeric_features),
            ('cat_pipeline', Pipeline([
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), self.categorical_features)
        ])
    
    def process_labels(self, y: pd.Series, is_train: bool) -> (pd.Series, pd.Index):
        y = y.astype(str).str.strip().str.lower()  # !!! 统一小写
        
        # !!! 三分类映射规则
        replacement_map = {
            # 保留类别
            'normal': 'normal',
            'generic': 'generic_attack',
            'dos': 'generic_attack',
            'fuzzers': 'generic_attack',
            'reconnaissance': 'generic_attack',
            'exploits': 'exploit_attack',
            'shellcode': 'exploit_attack',
            'backdoor': 'exploit_attack',
            # 过滤类别
            'analysis': 'DROP',
            'worms': 'DROP',
            'suspicious': 'DROP',
            'backdoors': 'DROP'
        }
        y_processed = y.replace(replacement_map)
        
        ## 统一过滤逻辑
        mask = (y_processed != 'DROP') & (y_processed.notna())
        y_filtered = y_processed[mask].copy()
        
        # 修复点：动态获取训练集类别
        if is_train:
            # 根据实际数据确定类别
            self.label_categories_ = sorted(y_filtered.unique().tolist())  
            logging.info(f"训练类别确定: {self.label_categories_}")
        else:
            # 过滤测试集未出现类别
            y_filtered = y_filtered[y_filtered.isin(self.label_categories_)]

        # 返回处理后的标签和有效索引
        y_processed_series = pd.Series(
            pd.Categorical(y_filtered, categories=self.label_categories_),
            index=y_filtered.index,
            name=y.name
        )
        return y_processed_series, y_filtered.index  # 返回有效索引

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:

        # 新增：验证原始字段完整性
        required_original = ['ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
        missing = [col for col in required_original if col not in X.columns]
        if missing:
            raise ValueError(f"原始数据缺失关键字段: {missing}")

        """修正后的fit_transform方法"""
        # 特征生成
        X_eng = self._advanced_feature_engineering(X)
        y_processed, valid_indices = self.process_labels(y, is_train=True)  # 获取有效索引

        # 根据有效索引同步过滤特征和标签
        X_eng = X_eng.loc[valid_indices]  # 关键修复：过滤X
        y_processed = y_processed.loc[valid_indices]

        # 新增：打印特征生成后的列信息
        print("生成的特征列:", X_eng.columns.tolist())

        # 确保数值列类型正确（关键修复点）
        numeric_cols = X_eng.select_dtypes(include=np.number).columns
        X_eng[numeric_cols] = X_eng[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 填充可能的NaN值
        X_eng[numeric_cols] = X_eng[numeric_cols].fillna(0)
        # 强制转换数据类型
        for col in self.categorical_features:
            X_eng[col] = X_eng[col].astype('category').cat.add_categories('missing')
        X_eng[self.categorical_features] = X_eng[self.categorical_features].fillna('missing')
        
        # 确定特征类型
        self.numeric_features = X_eng.select_dtypes(include=np.number).columns[
                    X_eng.select_dtypes(include=np.number).var() >= 0
                ].tolist()
        self.numeric_features = [f for f in self.numeric_features if f not in self.categorical_features]
        
        # 构建处理管道（添加异常值处理）
        self.column_transformer = self._build_preprocessing_pipeline()
        try:
            X_processed = self.column_transformer.fit_transform(X_eng, y_processed)
        except ValueError as e:
            print("预处理异常，当前数值特征：", self.numeric_features)
            print("样本数据：\n", X_eng[self.numeric_features].describe())
            raise e
        
        # 检查处理结果有效性
        if X_processed.shape[1] == 0:
            X_processed = np.zeros((len(X_eng), 1))  # 兜底方案
            raise ValueError(
                f"预处理后无有效特征！输入形状{X_eng.shape} "
                f"数值特征：{self.numeric_features} 类别特征：{self.categorical_features}"
            )
        
        # 特征选择优化
        self.feature_selector = SelectKBest(mutual_info_classif, k='all')
        self.feature_selector.fit(X_processed, y_processed.cat.codes)
        scores = self.feature_selector.scores_
        
        # 修改特征选择逻辑
        # 动态调整k值
        valid_features = np.where(scores > 0)[0]
        k = max(1, len(valid_features))
        
        # 确保不超过实际特征数
        k = min(k, X_processed.shape[1])
        self.feature_selector.set_params(k=k)
        X_selected = self.feature_selector.transform(X_processed)
        
        # 最终检查
        if X_selected.shape[1] == 0:
            # 保留至少一个特征作为兜底
            X_selected = X_processed[:, :1]  
            self.logger.warning("特征选择失败，强制保留首列特征")

        return X_selected, y_processed.cat.codes

    # 修改transform方法
    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        # 特征处理
        X_trans = self._advanced_feature_engineering(X)
        y_processed, valid_indices = self.process_labels(y, is_train=False)  # 首次处理

        # 同步过滤特征和标签
        X_trans = X_trans.loc[valid_indices]
        y_processed = y_processed.loc[valid_indices]

        # 预处理流程
        X_processed = self.column_transformer.transform(X_trans)
        X_selected = self.feature_selector.transform(X_processed)

        # 直接使用已处理的y_processed，无需再次处理
        y_filtered = y_processed.reset_index(drop=True)  # 确保索引连续

        return X_selected, y_filtered.cat.codes.values.astype(int)

    def _feature_engineering(self, X, is_train):
        # 确保数据类型正确
        for col in ['proto', 'service', 'state']:
            X[col] = X[col].astype('category')
            if is_train:
                self.cat_mapping[col] = X[col].cat.categories.tolist()
            X[col] = X[col].cat.codes.astype('int16')
        return X

    def validate(self):
        assert self.label_categories_ == ['normal', 'generic_attack', 'exploit_attack'], "标签类别错误"

        
    def _handle_zero_division(self, numerator, denominator):
        """安全除法处理"""
        return np.divide(numerator, denominator, 
                        out=np.zeros_like(numerator, dtype=np.float32),
                        where=(denominator != 0))

# ==================== 评估器 ====================
class EnhancedEvaluator:
    @staticmethod
    def log_parameter_update(iteration, params, scores, logger, memory_guard=None):
        """随机搜索参数轨迹记录"""
        if memory_guard is None:
            memory_guard = AdvancedMemoryGuard()
        stats = memory_guard.get_cpu_stats()
        param_str = ' | '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' 
                             for k,v in params.items()])
        log_entry = (
            f"迭代{iteration:03d}/{RS_CONFIG['n_iter']} "
            f"| {param_str} "
            f"| 分数:{np.mean(scores):.4f}±{np.std(scores):.4f} "
            f"| 内存:{psutil.Process().memory_info().rss/1024**3:.2f}GB "
            f"| CPU:{stats['proc_peak']:.1f}%"
        )
        logger.info(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] {log_entry}")

    @staticmethod
    def save_best_params(optimizer, logger, memory_guard, start_time):
        """保存完整训练日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV格式的完整交叉验证结果
        cv_results = pd.DataFrame(optimizer.cv_results_)
        csv_path = os.path.join(LOG_DIR, f'full_results_{timestamp}.csv')
        cv_results.to_csv(csv_path, index=False)
        
        # 生成详细日志内容
        log_content = f"实验时间: {timestamp}\n"
        log_content += f"最佳参数: {optimizer.best_params_}\n\n"
        
        # 添加系统指标
        process = psutil.Process()
        stats = memory_guard.get_cpu_stats()
        log_content += f"内存使用峰值: {process.memory_info().rss/1024**3:.2f}GB\n"
        log_content += f"CPU使用峰值: {stats['proc_peak']:.1f}%\n\n"
        
        # 添加交叉验证结果摘要
        log_content += "=== 交叉验证结果摘要 ===\n"
        log_content += f"最佳分数: {optimizer.best_score_:.4f}\n"
        log_content += f"参数组合总数: {len(cv_results)}\n"
        log_content += f"平均拟合时间: {cv_results['mean_fit_time'].mean():.2f}s\n"
        
        # 参数轨迹记录
        log_content += "\n=== 参数轨迹 ===\n"
        # 同步参数轨迹格式（与BO_SVM保持一致）
        for i, (params, score) in enumerate(zip(optimizer.cv_results_['params'], optimizer.cv_results_['mean_test_score'])):
            param_str = ' | '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' 
                             for k,v in params.items()])
            log_content += f'迭代{i+1:03d}/{RS_CONFIG["n_iter"]} | {param_str} | 分数:{score:.4f}\n'

        # 添加memory_guard统计
        # 统一资源统计方式（参考BO_SVM实现）
        duration = time.time() - start_time if 'start_time' in locals() else 0
        stats = memory_guard.get_cpu_stats()
        log_content += f"\n=== 系统资源统计 ===\n"
        log_content += f"训练耗时: {duration:.2f}秒\n"
        log_content += f"内存峰值: {stats['peak_mem']/1024**3:.2f}GB\n"
        log_content += f"CPU峰值（进程）: {stats['proc_peak']:.1f}%\n"
        log_content += f"CPU峰值（系统）: {stats['sys_peak']:.1f}%\n"
        
        # 保存日志文件
        log_path = os.path.join(LOG_DIR, f'full_log_{timestamp}.log')
        with open(log_path, 'w') as f:
            f.write(log_content)
        
        logger.info(f"完整日志已保存至: {log_path}")
        logger.info(f"CSV格式结果已保存至: {csv_path}")
        # 输出最优参数
        logger.info('\n最佳模型参数:')
        for param, value in optimizer.best_params_.items():
            logger.info(f' - {param}: {value}')
        return log_path
    


    @staticmethod
    def evaluate(y_true, y_pred, train_time, infer_time):
        # !!! 类别过滤
        valid_classes = ['normal', 'generic_attack', 'exploit_attack']
        mask = y_true.isin(valid_classes)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # 预测结果修正
        y_pred = y_pred.where(y_pred.isin(valid_classes), 'normal')
        
        sample_stats = {
            '测试样本数': len(y_true),
            '标签分布': y_true.value_counts().to_dict()
        }
        
        cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
        
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tn = cm.sum() - (fp + fn + np.diag(cm))
        fpr = np.nanmean(fp / (fp + tn))
        fnr = np.nanmean(fn / (fn + np.diag(cm)))
        
        report = classification_report(
            y_true, y_pred, 
            labels=valid_classes,
            target_names=valid_classes,
            zero_division=0
        )
        
        return {
            'Weighted F1': round(f1, 4),
            'MCC': round(mcc, 4),
            'FPR': round(fpr, 4),
            'FNR': round(fnr, 4),
            'Train Time (s)': round(train_time, 1),
            'Inference Time (s)': round(infer_time, 1),
            'Classification Report': report,
            'Confusion Matrix': cm
        }

# ==================== 训练流程 ====================
def main():
    logger = setup_logger()
    mem_guard = AdvancedMemoryGuard()
    
    try:
        # 数据加载
        logger.info("\n=== 数据加载 ===")
        train_data = pd.read_csv(
            os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
            dtype=UNSW_FEATURES
        ).sample(50000, random_state=42)

        # 数据预处理
        logger.info("\n=== 数据预处理 ===")
        start_preprocess = time.time()
        preprocessor = EnhancedPreprocessor()
        X_train, y_train = preprocessor.fit_transform(
            train_data.drop(columns=['attack_cat']), 
            train_data['attack_cat']
        )
        logger.info(f"有效样本数: {len(X_train)}, 预处理耗时: {time.time()-start_preprocess:.1f}s")

        # 初始化随机搜索
        rs = RandomizedSearchCV(
            estimator=SVC(kernel='rbf', cache_size=2000),
            param_distributions=SVM_PARAM_DIST,
            **RS_CONFIG
        )

        # 执行搜索
        logger.info("\n=== 开始随机搜索 ===") 
        start_search = time.time()
        rs.fit(X_train, y_train)
        search_duration = time.time() - start_search
        logger.info(f"随机搜索完成! 耗时: {search_duration:.1f}s")

        # 保存最佳模型
        joblib.dump({
            'model': rs.best_estimator_,
            'params': rs.best_params_,
            'preprocessor': preprocessor,
            'classes': preprocessor.label_categories_
        }, MODEL_SAVE_PATH)
        logger.info(f"\n=== 模型保存成功 ===")
        EnhancedEvaluator.save_best_params(rs, logger, mem_guard, start_search)

        # 执行增强可视化分析
        monitor_data = {
            'iterations': list(range(len(rs.cv_results_['mean_fit_time']))),
            'scores': rs.cv_results_['mean_test_score'],
            'memory': [psutil.virtual_memory().percent for _ in rs.cv_results_['params']],
            'cpu': [psutil.cpu_percent() for _ in rs.cv_results_['params']]
        }

        # 测试评估
        logger.info("\n=== 测试评估 ===")
        test_data = pd.read_csv(
            os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"),
            dtype=UNSW_FEATURES
        )
        X_test, y_test = preprocessor.transform(
            test_data.drop(columns=['attack_cat']),
            test_data['attack_cat']
        )
        
        start_infer = time.time()
        y_pred = rs.predict(X_test)
        infer_time = time.time() - start_infer

        # 转换预测标签
        y_test_labels = np.take(preprocessor.label_categories_, y_test)
        y_pred_labels = np.take(preprocessor.label_categories_, y_pred.astype(int))

        metrics = EnhancedEvaluator.evaluate(
            pd.Series(y_test_labels),
            pd.Series(y_pred_labels),
            rs.cv_results_['mean_fit_time'][-1],
            infer_time
        )
        logger.info(f"测试结果:\n{pd.Series(metrics).to_string()}\n分类报告:\n{metrics['Classification Report']}")

    except Exception as e:
        logger.error(f"运行异常: {str(e)}", exc_info=True)
    finally:
        mem_guard.stop()  # 停止监控线程
        # 获取最终资源统计
        cpu_stats = mem_guard.get_cpu_stats()
        logger.info(
            "\n=== 资源使用总结 ===\n"
            f"峰值内存: {mem_guard.peak_mem/1024**3:.2f}GB\n"
            f"系统CPU - 平均: {cpu_stats['sys_avg']:.1f}% | 峰值: {cpu_stats['sys_peak']:.1f}%\n"
            f"进程CPU - 平均: {cpu_stats['proc_avg']:.1f}% | 峰值: {cpu_stats['proc_peak']:.1f}%"
        )

def setup_logger():
    logger = logging.getLogger('IDS_RS_Training')
    logger.propagate = False
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(LOG_DIR, f"RS_SVM_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 清除可能存在的默认处理器
        if logging.root.handlers:
            logging.root.handlers = []
            
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    main()