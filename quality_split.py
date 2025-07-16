import pandas as pd
from sqlalchemy import create_engine
import logging
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuleEngine:
    def __init__(self):
        # 通用规则参数
        self.current_rules = {
            "min_stay_time_high": 10.0,    # 高质量最小停留时间
            "min_stay_time_mid": 5.0,      # 中质量最小停留时间
            "max_stay_time": 30.0,         # 最大停留时间阈值
            "success_rate_threshold": 0.8,  # 成功率阈值
            "score_high_threshold": 0.8,    # 高质量分数阈值
            "score_mid_threshold": 0.4,     # 中质量分数阈值
            "min_time_weight": 0.5,         # 最小时间权重
            "task_weights": {              # 不同任务类型的权重
                "default": 1.0,
                "kyc_verification": 1.2,   # KYC验证任务
                "address_analysis": 1.1,    # 地址分析
                "document_check": 0.9       # 文档检查
            }
        }
        # 初始化模型
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
        self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 注册任务验证器
        self.task_validators = {
            "kyc_verification": self.validate_kyc_result,
            "default": self.validate_general_result
        }

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        """安全更新规则参数"""
        try:
            # 特殊处理task_weights的更新
            if "task_weights" in new_rules:
                self.current_rules["task_weights"].update(new_rules["task_weights"])
                del new_rules["task_weights"]
            
            self.current_rules.update(new_rules)
            logger.info(f"规则参数已更新: {new_rules}")
        except Exception as e:
            logger.error(f"规则更新失败: {e}")
            raise

    def validate_kyc_result(self, row: pd.Series) -> bool:
        """KYC验证结果校验"""
        try:
            result = json.loads(row["result_info"])
            required_fields = ["verification_status", "confidence_score", "document_type"]
            return (
                all(field in result for field in required_fields) and
                isinstance(result["confidence_score"], (int, float)) and
                0 <= result["confidence_score"] <= 1
            )
        except:
            return False

    def validate_general_result(self, row: pd.Series) -> bool:
        """通用结果验证方法"""
        try:
            if pd.isna(row["result_info"]) or not row["result_info"]:
                return False
                
            # 尝试解析JSON
            try:
                result = json.loads(row["result_info"])
                return bool(result)  # 非空字典或列表
            except json.JSONDecodeError:
                # 如果不是JSON，检查是否为有效字符串
                return isinstance(row["result_info"], str) and len(row["result_info"]) > 10
        except:
            return False

def fetch_data_from_db(
    host: str, 
    port: int, 
    user: str, 
    password: str, 
    database: str, 
    table_name: str, 
    fields: list,
    days: int = 30
) -> pd.DataFrame:
    """从数据库获取最近N天的数据"""
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
        
        # 确保包含必要字段
        required_fields = ["prompt", "result_info", "create_time"]
        for field in required_fields:
            if field not in fields:
                fields.append(field)
                
        # 获取最近数据
        recent_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = f"""
        SELECT {', '.join(fields)} 
        FROM {table_name} 
        WHERE create_time >= '{recent_date}' 
          AND prompt IS NOT NULL 
          AND prompt != ''
          AND result_info IS NOT NULL
          AND result_info != ''
        """
        
        df = pd.read_sql(query, engine)
        logger.info(f"成功获取 {len(df)} 条数据，时间范围: {recent_date} 至今")
        return df
    except Exception as e:
        logger.error(f"数据库查询失败: {e}")
        raise
    finally:
        if 'engine' in locals():
            engine.dispose()

def compute_features(df: pd.DataFrame, rule_engine: RuleEngine) -> pd.DataFrame:
    """计算特征"""
    df = df.copy()
    
    # 基础数据清洗
    df = df[df["prompt"].str.len() > 10]  # 过滤过短prompt
    if "stay_time" in df.columns:
        df = df[df["stay_time"] > 0]  # 过滤无效停留时间
    
    # 结果有效性验证
    if "branch" in df.columns:
        df["result_valid"] = df.apply(
            lambda row: rule_engine.task_validators.get(
                row["branch"], 
                rule_engine.validate_general_result
            )(row),
            axis=1
        ).astype(int)
    else:
        df["result_valid"] = df.apply(rule_engine.validate_general_result, axis=1).astype(int)
    
    # 时间相关特征
    if "create_time" in df.columns:
        df["create_time"] = pd.to_datetime(df["create_time"])
        df["time_weight"] = np.exp(-(datetime.now() - df["create_time"]).dt.total_seconds() / (7 * 24 * 3600))
    else:
        df["time_weight"] = 1.0
    
    # 用户交互特征
    if "user_id" in df.columns:
        df["interaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    else:
        df["interaction_count"] = 1
    
    # 文本特征
    df["prompt_length"] = df["prompt"].str.len()
    df["word_count"] = df["prompt"].str.split().str.len()
    
    # 任务权重
    if "branch" in df.columns:
        df["task_weight"] = df["branch"].map(
            lambda x: rule_engine.current_rules["task_weights"].get(x, 1.0))
    else:
        df["task_weight"] = 1.0
    
    return df

def split_data(df: pd.DataFrame, rule_engine: RuleEngine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """数据分流"""
    try:
        df = compute_features(df, rule_engine)
        
        # 确保必要字段存在
        for col in ["stay_time", "is_quality", "result_valid"]:
            if col not in df.columns:
                df[col] = 0 if col != "is_quality" else 1
        
        # 计算综合评分
        max_stay_time = rule_engine.current_rules["max_stay_time"]
        df["score"] = (
            0.4 * (df["is_quality"] + 1) / 2 +
            0.3 * (df["stay_time"] / max_stay_time).clip(0, 1) +
            0.2 * df["result_valid"] +
            0.1 * df["time_weight"]
        )
        
        # 动态阈值
        score_high = df["score"].quantile(0.75)
        score_mid = df["score"].quantile(0.5)
        time_high = df["stay_time"].quantile(0.75)
        
        # 分流
        high_quality = df[
            (df["score"] >= score_high) &
            (df["stay_time"] >= time_high) &
            (df["result_valid"] == 1)
        ]
        
        mid_quality = df[
            (df["score"] >= score_mid) &
            (df["result_valid"] == 1) &
            (~df.index.isin(high_quality.index))
        ]
        
        low_quality = df[
            ~df.index.isin(high_quality.index) & 
            ~df.index.isin(mid_quality.index)
        ]
        
        return high_quality, mid_quality, low_quality
    except Exception as e:
        logger.error(f"数据分流失败: {e}")
        raise

def process_data(df: pd.DataFrame, rule_engine: RuleEngine) -> Dict[str, Any]:
    """处理数据并生成新规则"""
    try:
        if df.empty:
            logger.warning("输入数据为空，返回默认规则")
            return rule_engine.current_rules.copy()
        
        # 计算特征
        df = compute_features(df, rule_engine)
        
        # 训练模型
        features = ["stay_time", "is_quality", "result_valid", "time_weight"]
        if "interaction_count" in df.columns:
            features.append("interaction_count")
        rule_engine.train_decision_tree(df, features)
        
        # 生成新规则
        new_rules = rule_engine.current_rules.copy()
        
        # 更新停留时间阈值
        if "stay_time" in df.columns:
            new_rules.update({
                "min_stay_time_high": df["stay_time"].quantile(0.75),
                "min_stay_time_mid": df["stay_time"].quantile(0.5),
                "max_stay_time": df["stay_time"].quantile(0.95) * 1.5
            })
        
        # 更新任务权重
        if "branch" in df.columns and "is_quality" in df.columns:
            task_quality = df.groupby("branch")["is_quality"].mean()
            for task, quality in task_quality.items():
                new_rules["task_weights"][task] = min(1.5, max(0.7, quality * 2))
        
        return new_rules
    except Exception as e:
        logger.error(f"规则生成失败: {e}")
        return rule_engine.current_rules.copy()

def save_to_finetune(df: pd.DataFrame, output_path: str):
    """保存高质量数据"""
    try:
        cols_to_save = ["prompt", "result_info"]
        df[cols_to_save].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条高质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存高质量数据失败: {e}")
        raise

def save_to_rl(df: pd.DataFrame, output_path: str):
    """保存中质量数据"""
    try:
        cols_to_save = ["prompt", "result_info"]
        df[cols_to_save].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条中质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存中质量数据失败: {e}")
        raise