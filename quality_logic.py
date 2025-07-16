import pandas as pd
from sqlalchemy import create_engine # 虽然在这个文件里不再直接用，但为了类型提示兼容性保留
import logging
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sentence_transformers import SentenceTransformer
import json

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

    # 实现了 train_decision_tree 方法，用于辅助规则生成
    def train_decision_tree(self, df: pd.DataFrame, features: List[str]):
        """训练决策树/随机森林模型来辅助规则生成。"""
        if 'is_quality' not in df.columns:
            logger.warning("训练模型缺少 'is_quality' 目标列，跳过训练。")
            return

        available_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if not available_features:
            logger.warning("没有可用的数值特征来训练模型，跳过训练。")
            return

        X = df[available_features].fillna(0) # 填充 NaN 值，简单的填充为0
        y = df['is_quality'].replace(-1, 0) # 将 -1 (负反馈) 转换为 0

        # 确保有足够数据且目标变量有多个类别
        if len(X) > 0 and len(np.unique(y)) > 1:
            try:
                self.clf.fit(X, y)
                logger.info(f"随机森林模型训练完成，使用了特征: {available_features}")
            except Exception as e:
                logger.error(f"训练随机森林模型失败: {e}", exc_info=True)
        else:
            logger.warning("数据不足或目标变量类别单一，无法训练随机森林模型。")


    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        """安全更新规则参数"""
        try:
            if "task_weights" in new_rules:
                self.current_rules["task_weights"].update(new_rules["task_weights"])
                del new_rules["task_weights"]

            self.current_rules.update(new_rules)
            logger.info(f"规则参数已更新: {new_rules}")
        except Exception as e:
            logger.error(f"规则更新失败: {e}", exc_info=True)
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


def compute_features(df: pd.DataFrame, rule_engine: RuleEngine) -> pd.DataFrame:
    """
    计算特征。
    此函数会接收一个 DataFrame，其中包含数据库的原始字段。
    在此阶段，不涉及 MinIO 的图片数据。
    """
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

    # --- 暂时不添加基于 'file_url' 或图片数据的特征 ---
    # 确保 'file_url' 存在，即使不用于计算特征，它也会被传递到后续阶段
    if "file_url" not in df.columns:
        df["file_url"] = None # 如果不存在，添加为 None，避免后续操作报错

    return df


def split_data(df: pd.DataFrame, rule_engine: RuleEngine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """数据分流"""
    try:
        df = compute_features(df, rule_engine) # compute_features 确保了特征计算

        # 确保必要字段存在
        # is_quality 的值在数据库中可能是 1 或 -1，这里统一为 0 或 1
        df['is_quality_binary'] = df['is_quality'].apply(lambda x: 1 if x == 1 else 0)


        # 计算综合评分 (这里不使用图片相关特征)
        max_stay_time = rule_engine.current_rules["max_stay_time"]
        df["score"] = (
            0.4 * df["is_quality_binary"] + # 直接使用二值化的is_quality
            0.3 * (df["stay_time"] / max_stay_time).clip(0, 1) +
            0.2 * df["result_valid"] +
            0.1 * df["time_weight"]
        )

        # 动态阈值
        score_high = df["score"].quantile(0.75)
        score_mid = df["score"].quantile(0.5)
        time_high = df["stay_time"].quantile(0.75)

        # 分流逻辑
        high_quality = df[
            (df["score"] >= score_high) &
            (df["stay_time"] >= time_high) &
            (df["result_valid"] == 1)
        ]

        mid_quality = df[
            (df["score"] >= score_mid) &
            (df["result_valid"] == 1) &
            (~df.index.isin(high_quality.index)) # 排除已经分到高质量的数据
        ]

        # 低质量数据是剩余的数据
        low_quality = df[
            ~df.index.isin(high_quality.index) &
            ~df.index.isin(mid_quality.index)
        ]

        return high_quality, mid_quality, low_quality
    except Exception as e:
        logger.error(f"数据分流失败: {e}", exc_info=True)
        raise

def process_data(df: pd.DataFrame, rule_engine: RuleEngine) -> Dict[str, Any]:
    """处理数据并生成新规则"""
    try:
        if df.empty:
            logger.warning("输入数据为空，返回默认规则")
            return rule_engine.current_rules.copy()

        # 计算特征
        df = compute_features(df, rule_engine)

        # 训练模型 (确保 RuleEngine 中有 train_decision_tree 方法)
        # 训练特征不包括 MinIO 相关列
        features = ["stay_time", "result_valid", "time_weight", "prompt_length", "word_count", "task_weight"]
        if "interaction_count" in df.columns:
            features.append("interaction_count")

        rule_engine.train_decision_tree(df, features) # 调用 RuleEngine 实例的方法

        # 生成新规则
        new_rules = rule_engine.current_rules.copy()

        # 更新停留时间阈值
        if "stay_time" in df.columns and not df["stay_time"].empty:
            new_rules.update({
                "min_stay_time_high": df["stay_time"].quantile(0.75),
                "min_stay_time_mid": df["stay_time"].quantile(0.5),
                "max_stay_time": df["stay_time"].quantile(0.95) * 1.5
            })
            logger.info("更新了停留时间阈值。")
        else:
            logger.warning("无法根据 'stay_time' 列更新停留时间阈值，可能为空或不存在。")


        # 更新任务权重
        if "branch" in df.columns and "is_quality" in df.columns and not df["branch"].empty:
            # is_quality 转换为 0/1 参与平均计算
            df['is_quality_binary'] = df['is_quality'].apply(lambda x: 1 if x == 1 else 0)
            task_quality = df.groupby("branch")["is_quality_binary"].mean()
            for task, quality in task_quality.items():
                new_rules["task_weights"][task] = min(1.5, max(0.7, quality * 2)) # 将质量映射到权重范围
            logger.info("更新了任务权重。")
        else:
            logger.warning("无法根据 'branch' 或 'is_quality' 列更新任务权重，可能为空或不存在。")

        return new_rules
    except Exception as e:
        logger.error(f"规则生成失败: {e}", exc_info=True)
        return rule_engine.current_rules.copy() # 失败时返回旧规则

def save_to_finetune(df: pd.DataFrame, output_path: str):
    """保存高质量数据"""
    try:
        # 保存时依然包含 'file_url'，因为它是原始数据的一部分
        # 但不包含 'image_data_from_minio'，因为它现在不参与质量分流，且可能内存占用大
        cols_to_save = ["prompt", "result_info", "file_url", "branch", "is_quality", "stay_time"]
        # 确保只保存 DataFrame 中实际存在的列
        actual_cols = [col for col in cols_to_save if col in df.columns]
        df[actual_cols].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条高质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存高质量数据失败: {e}", exc_info=True)
        raise

def save_to_rl(df: pd.DataFrame, output_path: str):
    """保存中质量数据"""
    try:
        cols_to_save = ["prompt", "result_info", "file_url", "branch", "is_quality", "stay_time"]
        actual_cols = [col for col in cols_to_save if col in df.columns]
        df[actual_cols].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条中质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存中质量数据失败: {e}", exc_info=True)
        raise