import pandas as pd
import logging
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import io
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)


LOCAL_CLIP_MODEL_PATH = "models/clip-vit-base-patch32"

_clip_processor = None
_clip_model = None
_clip_projection_dim: Optional[int] = None

def _load_clip_model():
    global _clip_processor, _clip_model, _clip_projection_dim
    if _clip_processor is None or _clip_model is None:
        try:
            logger.info(f"正在从本地路径 '{LOCAL_CLIP_MODEL_PATH}' 加载 CLIP 视觉模型和处理器...")
            _clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH)
            _clip_model = CLIPModel.from_pretrained(LOCAL_CLIP_MODEL_PATH)
            _clip_model.eval()
            _clip_projection_dim = _clip_model.config.projection_dim
            logger.info(f"CLIP 模型本地加载完成，特征维度: {_clip_projection_dim}。")
        except Exception as e:
            logger.error(f"从本地路径 '{LOCAL_CLIP_MODEL_PATH}' 加载 CLIP 模型失败: {e}", exc_info=True)
            logger.warning("尝试从 Hugging Face 仓库重新下载...")
            try:
                _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model.eval()
                _clip_projection_dim = _clip_model.config.projection_dim
                logger.info(f"CLIP 模型从网络下载并加载完成，特征维度: {_clip_projection_dim}。")
            except Exception as e_fallback:
                logger.error(f"从网络下载和加载 CLIP 模型也失败: {e_fallback}", exc_info=True)
                _clip_processor = None
                _clip_model = None
                _clip_projection_dim = 512 # Default dimension if all else fails

class RuleEngine:
    def __init__(self):
        self.current_rules = {
            "min_stay_time_high": 10.0,
            "min_stay_time_mid": 5.0,
            "max_stay_time": 30.0,
            "success_rate_threshold": 0.8,
            "score_high_threshold": 0.8,
            "score_mid_threshold": 0.4,
            "min_time_weight": 0.5,
            "task_weights": {
                "default": 1.0,
                "kyc_verification": 1.2,
                "address_analysis": 1.1,
                "document_check": 0.9
            }
        }
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
        self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        _load_clip_model() # Ensure CLIP model is loaded once on RuleEngine instantiation
        self.task_validators = {
            "kyc_verification": self.validate_kyc_result,
            "default": self.validate_general_result
        }

    def train_decision_tree(self, df: pd.DataFrame, features: List[str]):
        if 'is_quality' not in df.columns:
            logger.warning("训练模型缺少 'is_quality' 目标列，跳过训练。")
            return

        y = df['is_quality'].replace(-1, 0)
        if len(np.unique(y)) <= 1:
            logger.warning("目标变量 'is_quality' 类别单一，无法训练随机森林模型。")
            return

        available_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if not available_features:
            logger.warning("没有可用的数值特征来训练模型，跳过训练。")
            return

        X = df[available_features].fillna(0)
        
        if X.empty:
            logger.warning("训练特征矩阵 X 为空，无法训练随机森林模型。")
            return

        try:
            self.clf.fit(X, y)
            logger.info(f"随机森林模型训练完成，使用了特征: {available_features}")
        except Exception as e:
            logger.error(f"训练随机森林模型失败: {e}", exc_info=True)


    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        try:
            if "task_weights" in new_rules:
                self.current_rules["task_weights"].update(new_rules["task_weights"])
                del new_rules["task_weights"]

            self.current_rules.update(new_rules)
            logger.info(f"规则参数已更新: { {k: v for k, v in new_rules.items() if k != 'task_weights'} }")
            if "task_weights" in new_rules:
                 logger.info(f"任务权重已更新: {new_rules['task_weights']}")

        except Exception as e:
            logger.error(f"规则更新失败: {e}", exc_info=True)
            raise

    def validate_kyc_result(self, row: pd.Series) -> bool:
        try:
            result = json.loads(str(row["result_info"]))
            required_fields = ["verification_status", "confidence_score", "document_type"]
            return (
                all(field in result for field in required_fields) and
                isinstance(result["confidence_score"], (int, float)) and
                0 <= result["confidence_score"] <= 1
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return False
        except Exception as e:
            logger.error(f"KYC结果验证未知错误: {e}", exc_info=True)
            return False

    def validate_general_result(self, row: pd.Series) -> bool:
        try:
            if pd.isna(row["result_info"]) or not str(row["result_info"]).strip():
                return False
            try:
                result = json.loads(str(row["result_info"]))
                return bool(result)
            except json.JSONDecodeError:
                return isinstance(row["result_info"], str) and len(str(row["result_info"]).strip()) > 10
        except Exception as e:
            logger.error(f"通用结果验证未知错误: {e}", exc_info=True)
            return False

def compute_features(df: pd.DataFrame, rule_engine: RuleEngine) -> pd.DataFrame:
    """
    计算特征。
    此函数现在将从 'image_raw_data_list' 列中提取图片特征。
    """
    df = df.copy()
    initial_rows = len(df)
    logger.debug(f"compute_features: 初始 DataFrame 行数: {initial_rows}")
    new_cols_data = {}

    # --- 关键修改：处理 stay_time 列 ---
    if "stay_time" in df.columns:
        logger.debug(f"compute_features: 原始 'stay_time' 列数据类型: {df['stay_time'].dtype}")
        logger.debug(f"compute_features: 原始 'stay_time' 列前5个值: {df['stay_time'].head().tolist()}")
        
        # 1. 强制转换为数值类型，无法转换的变为 NaN
        df['stay_time'] = pd.to_numeric(df['stay_time'], errors='coerce')
        logger.debug(f"compute_features: 转换为数值型后 'stay_time' 列中 NaN 数量: {df['stay_time'].isnull().sum()}")

        # 2. 将 NaN 填充为一个默认的有效值 (例如 1，因为要求不能为 0 或负数)
        #    并处理小于等于 0 的值，也将其填充为这个默认有效值。
        #    这里的 1 是一个示例，你可以根据业务需求选择其他合适的最小值。
        default_min_stay_time = 1.0 # 默认的最小有效停留时间，确保不为0或负数
        df['stay_time'] = df['stay_time'].fillna(default_min_stay_time)
        df['stay_time'] = df['stay_time'].apply(lambda x: max(x, default_min_stay_time)) # 确保所有值都 >= default_min_stay_time
        
        logger.debug(f"compute_features: 填充和清理后 'stay_time' 列前5个值: {df['stay_time'].head().tolist()}")
        logger.debug(f"compute_features: 填充和清理后 'stay_time' 列中 <= 0 的值数量: {(df['stay_time'] <= 0).sum()}") # 此时应为 0

    else:
        logger.warning("DataFrame 中缺少 'stay_time' 列。为确保后续计算，将创建并填充为 1.0。")
        df['stay_time'] = 1.0 # 如果没有 stay_time 列，也给一个默认值


    # --- 核心数据完整性过滤：prompt 和 result_info (这部分保持不变) ---
    # 1. prompt 长度过滤
    if "prompt" in df.columns:
        original_prompt_rows = len(df)
        df = df[df["prompt"].astype(str).str.len() > 10]
        logger.debug(f"compute_features: 过滤 prompt 长度 > 10 后剩余行数: {len(df)} (减少了 {original_prompt_rows - len(df)} 行)")
    else:
        logger.warning("DataFrame 中缺少 'prompt' 列，跳过 prompt 长度过滤。")

    # 2. result_info 过滤 (新增或修改)
    if "result_info" in df.columns:
        original_result_info_rows = len(df)
        df = df[df["result_info"].notna() & (df["result_info"].astype(str).str.strip().str.len() > 0)]
        logger.debug(f"compute_features: 过滤 result_info 非空后剩余行数: {len(df)} (减少了 {original_result_info_rows - len(df)} 行)")
    else:
        logger.warning("DataFrame 中缺少 'result_info' 列，跳过 result_info 过滤。这可能会影响数据质量。")

    # If df becomes empty after these filters, return early
    if df.empty:
        logger.warning("DataFrame 在特征计算的早期核心过滤后变为空，返回空 DataFrame。")
        current_clip_dim = _clip_projection_dim if _clip_projection_dim is not None else 512
        for i in range(current_clip_dim):
            new_cols_data[f"image_feature_{i}"] = []
        if "has_image_data" in df.columns:
            new_cols_data["has_image_data"] = []
        else:
            new_cols_data["has_image_data"] = []
        return pd.DataFrame(new_cols_data, index=df.index)

    if "branch" in df.columns:
        new_cols_data["result_valid"] = df.apply(
            lambda row: rule_engine.task_validators.get(
                row["branch"],
                rule_engine.validate_general_result
            )(row),
            axis=1
        ).astype(int)
    else:
        new_cols_data["result_valid"] = df.apply(rule_engine.validate_general_result, axis=1).astype(int)

    if "create_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["create_time"]):
        df["create_time"] = pd.to_datetime(df["create_time"], errors='coerce')
        df_valid_time = df[df["create_time"].notna()]
        
        if not df_valid_time.empty:
            time_diff_seconds = (datetime.now() - df_valid_time["create_time"]).dt.total_seconds()
            temp_time_weight = np.exp(-time_diff_seconds / (7 * 24 * 3600))
            new_cols_data["time_weight"] = pd.Series(1.0, index=df.index) # Initialize with default 1.0
            new_cols_data["time_weight"].loc[df_valid_time.index] = temp_time_weight
        else:
            new_cols_data["time_weight"] = 1.0
            logger.warning("所有 'create_time' 均为无效值，'time_weight' 将设置为 1.0。")
    else:
        new_cols_data["time_weight"] = 1.0
        logger.warning("DataFrame 中缺少 'create_time' 列或其类型不正确，'time_weight' 将设置为 1.0。")


    if "user_id" in df.columns and not df["user_id"].empty:
        new_cols_data["interaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    else:
        new_cols_data["interaction_count"] = 1
        logger.warning("DataFrame 中缺少 'user_id' 列或其为空，'interaction_count' 将设置为 1。")


    new_cols_data["prompt_length"] = df["prompt"].astype(str).str.len().fillna(0) # Ensure string conversion
    new_cols_data["word_count"] = df["prompt"].astype(str).str.split().str.len().fillna(0) # Ensure string conversion

    if "branch" in df.columns and not df["branch"].empty:
        new_cols_data["task_weight"] = df["branch"].map(
            lambda x: rule_engine.current_rules["task_weights"].get(x, 1.0))
    else:
        new_cols_data["task_weight"] = 1.0
        logger.warning("DataFrame 中缺少 'branch' 列或其为空，'task_weight' 将设置为 1.0。")


    # --- 图片特征提取 ---
    current_clip_dim = _clip_projection_dim if _clip_projection_dim is not None else 512

    if "image_raw_data_list" in df.columns and _clip_model is not None and _clip_processor is not None:
        logger.info("正在提取图片特征...")
        
        total_images_in_df = df['image_raw_data_list'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        successful_reads_in_df = df['image_read_success'].sum()
        logger.debug(f"进入 compute_features 图片处理阶段，df 中总图片数量: {total_images_in_df}，成功读取标记的记录数: {successful_reads_in_df}")
        logger.info(f"CLIP 模型的特征维度: {current_clip_dim}")

        image_features_list = []
        processed_image_count = 0

        for index, row in df.iterrows():
            record_image_features = []
            if row['image_read_success'] and isinstance(row['image_raw_data_list'], list) and row['image_raw_data_list']:
                logger.debug(f"Row {index} (file_url: {row.get('file_url', 'N/A')}): 'image_read_success' is True, 'image_raw_data_list' contains {len(row['image_raw_data_list'])} image(s).")
                
                # Check for image_object_keys_list existence
                image_object_keys_present = 'image_object_keys_list' in row and isinstance(row['image_object_keys_list'], list)

                for i, img_bytes in enumerate(row['image_raw_data_list']):
                    obj_key_for_log = f"N/A (index {i})"
                    if image_object_keys_present and i < len(row['image_object_keys_list']):
                        obj_key_for_log = row['image_object_keys_list'][i]

                    try:
                        if img_bytes is None:
                            logger.warning(f"Row {index}, Image {i}: MinIO object '{obj_key_for_log}' returned None bytes. Skipping.")
                            continue
                        if not isinstance(img_bytes, bytes):
                            logger.warning(f"Row {index}, Image {i}: MinIO object '{obj_key_for_log}' bytes is not of type 'bytes' ({type(img_bytes)}). Skipping.")
                            continue
                        if len(img_bytes) == 0:
                            logger.warning(f"Row {index}, Image {i}: MinIO object '{obj_key_for_log}' returned empty (0 bytes). Skipping.")
                            continue

                        logger.debug(f"Row {index}, Image {i}: Processing object '{obj_key_for_log}' with {len(img_bytes)} bytes.")

                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        inputs = _clip_processor(images=image, return_tensors="pt").to(_clip_model.device)
                        with torch.no_grad():
                            image_features = _clip_model.get_image_features(**inputs)
                        
                        record_image_features.append(image_features.squeeze().cpu().numpy())
                        processed_image_count += 1
                        logger.debug(f"Row {index}, Image {i}: Successfully extracted feature for '{obj_key_for_log}'.")
                    except UnidentifiedImageError:
                        logger.warning(f"Row {index}, Image {i}: UnidentifiedImageError for object '{obj_key_for_log}'. Bytes are not a recognizable image format. Skipping.", exc_info=True) # Add exc_info for more details
                    except Exception as e:
                        logger.error(f"Row {index}, Image {i}: Error processing object '{obj_key_for_log}' from file_url: {row.get('file_url', 'N/A')}. Error: {e}", exc_info=True)
            else:
                logger.debug(f"Row {index} (file_url: {row.get('file_url', 'N/A')}): 'image_read_success' is {row.get('image_read_success', 'N/A')} or 'image_raw_data_list' is empty/None/not list. No images to process for this row.")

            if record_image_features:
                image_features_list.append(np.mean(record_image_features, axis=0))
            else:
                image_features_list.append(np.zeros(current_clip_dim))

        image_feature_columns = [f"image_feature_{i}" for i in range(current_clip_dim)]
        
        if image_features_list:
            # Ensure the index matches the filtered df's index
            image_features_df = pd.DataFrame(image_features_list, index=df.index, columns=image_feature_columns)
            for col in image_feature_columns:
                new_cols_data[col] = image_features_df[col]
            logger.info(f"成功提取 {processed_image_count} 张图片的特征（对应 {len(image_features_list)} 条记录的平均特征），维度为 {current_clip_dim}。")
        else:
            logger.warning("没有图片特征被成功提取。所有图片特征列将填充零。")
            for col in image_feature_columns:
                new_cols_data[col] = np.zeros(len(df))
    else:
        logger.warning("未检测到 'image_raw_data_list' 列或 CLIP 模型未加载，跳过图片特征提取。所有图片特征列将填充零。")
        dummy_dim = _clip_projection_dim if _clip_projection_dim is not None else 512
        image_feature_columns = [f"image_feature_{i}" for i in range(dummy_dim)]
        for col in image_feature_columns:
            new_cols_data[col] = np.zeros(len(df))


    if "image_read_success" in df.columns:
        new_cols_data["has_image_data"] = df["image_read_success"].astype(int)
        logger.debug("Added 'has_image_data' feature based on MinIO read status.")
    else:
        # If 'image_read_success' column doesn't exist, assume no image data for these records
        new_cols_data["has_image_data"] = 0 

    new_features_df = pd.DataFrame(new_cols_data, index=df.index)
    
    df = pd.concat([df, new_features_df], axis=1)

    return df

def split_data(df: pd.DataFrame, rule_engine: RuleEngine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """数据分流"""
    try:
        df_processed = compute_features(df.copy(), rule_engine)

        if df_processed.empty:
            logger.warning("特征计算后 DataFrame 为空，无法进行数据分流。")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        split_new_cols_data = {}

        if 'is_quality' in df_processed.columns:
            split_new_cols_data['is_quality_binary'] = df_processed['is_quality'].apply(lambda x: 1 if x == 1 else 0)
        else:
            split_new_cols_data['is_quality_binary'] = 0
            logger.warning("DataFrame 中缺少 'is_quality' 列，'is_quality_binary' 将设置为 0。")

        # 将 stay_time 默认处理为 0 或合理值，而不是过滤掉整个行
        max_stay_time = rule_engine.current_rules["max_stay_time"]
        stay_time_series = df_processed["stay_time"].fillna(0).clip(lower=0) if "stay_time" in df_processed.columns else pd.Series(0.0, index=df_processed.index)
        
        result_valid_series = df_processed["result_valid"].fillna(0) if "result_valid" in df_processed.columns else pd.Series(0.0, index=df_processed.index)
        time_weight_series = df_processed["time_weight"].fillna(1.0) if "time_weight" in df_processed.columns else pd.Series(1.0, index=df_processed.index)
        has_image_data_series = df_processed["has_image_data"].fillna(0) if "has_image_data" in df_processed.columns else pd.Series(0.0, index=df_processed.index)

        split_new_cols_data["score"] = (
            0.4 * pd.Series(split_new_cols_data["is_quality_binary"], index=df_processed.index) +
            0.3 * (stay_time_series / max_stay_time).clip(0, 1) + # stay_time 现在可能包含 NaN 或 0
            0.2 * result_valid_series +
            0.1 * time_weight_series
        )

        split_new_cols_data["score"] += 0.05 * has_image_data_series

        df_final = pd.concat([df_processed, pd.DataFrame(split_new_cols_data, index=df_processed.index)], axis=1)

        if df_final["score"].empty or df_final["stay_time"].empty or df_final["score"].isnull().all() or df_final["stay_time"].isnull().all():
             logger.warning("分流时 'score' 或 'stay_time' 列为空或全为 NaN，无法计算分位数。返回空分流结果。")
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 分位数计算现在应该更稳健，因为 stay_time 不再强制过滤掉行
        score_high = df_final["score"].quantile(0.75) if not df_final["score"].isnull().all() else 0.8
        score_mid = df_final["score"].quantile(0.5) if not df_final["score"].isnull().all() else 0.4
        
        # 考虑到 stay_time 现在可能包含 0 或 NaN，需要更谨慎地处理
        # 如果 df_final["stay_time"] 仍然全为 NaN，则使用默认值
        valid_stay_times = df_final["stay_time"].dropna()
        if not valid_stay_times.empty:
            time_high = valid_stay_times.quantile(0.75) 
        else:
            time_high = rule_engine.current_rules["min_stay_time_high"]
            logger.warning("DataFrame 中所有 'stay_time' 均为 NaN，'time_high' 将使用默认规则。")


        high_quality = df_final[
            (df_final["score"] >= score_high) &
            (df_final["stay_time"] >= time_high) & # 这里继续使用 stay_time
            (df_final["result_valid"] == 1)
        ]

        mid_quality = df_final[
            (df_final["score"] >= score_mid) &
            (df_final["result_valid"] == 1) &
            (~df_final.index.isin(high_quality.index))
        ]

        low_quality = df_final[
            ~df_final.index.isin(high_quality.index) &
            ~df_final.index.isin(mid_quality.index)
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

        df_processed = compute_features(df.copy(), rule_engine)

        if df_processed.empty:
            logger.warning("特征计算后 DataFrame 为空，无法进行规则生成。返回默认规则。")
            return rule_engine.current_rules.copy()

        # Features list will now include image features regardless of their values (they will be 0 if no images)
        features = ["stay_time", "result_valid", "time_weight", "prompt_length", "word_count", "task_weight", "has_image_data"]
        if "interaction_count" in df_processed.columns:
            features.append("interaction_count")

        current_clip_dim = _clip_projection_dim if _clip_projection_dim is not None else 512
        image_feature_columns = [f"image_feature_{i}" for i in range(current_clip_dim)]
        # Filter for actual numeric columns available in df_processed
        features.extend([col for col in image_feature_columns if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])])
        
        # Ensure 'stay_time' is numeric for training, fillna(0) for safety
        if 'stay_time' in df_processed.columns:
            df_processed['stay_time'] = pd.to_numeric(df_processed['stay_time'], errors='coerce').fillna(0)
            
        rule_engine.train_decision_tree(df_processed, features)

        new_rules = rule_engine.current_rules.copy()

        # 调整规则更新逻辑，即使 stay_time 列有 NaN 或 0，只要有有效值就尝试更新
        if "stay_time" in df_processed.columns:
            # 只用非 NaN 和大于 0 的 stay_time 值来计算分位数
            valid_stay_times = df_processed["stay_time"][df_processed["stay_time"].notna() & (df_processed["stay_time"] > 0)]
            if not valid_stay_times.empty:
                new_rules.update({
                    "min_stay_time_high": valid_stay_times.quantile(0.75),
                    "min_stay_time_mid": valid_stay_times.quantile(0.5),
                    "max_stay_time": valid_stay_times.quantile(0.95) * 1.5
                })
                logger.info("更新了停留时间阈值。")
            else:
                logger.warning("所有 'stay_time' 值均为无效值，无法更新停留时间阈值，使用默认规则。")
        else:
            logger.warning("DataFrame 中缺少 'stay_time' 列，无法更新停留时间阈值。")


        if "branch" in df_processed.columns and "is_quality" in df_processed.columns \
           and not df_processed["branch"].empty and df_processed["is_quality"].notna().any() \
           and len(df_processed["branch"].unique()) > 0:
            
            df_processed['is_quality_binary'] = df_processed['is_quality'].apply(lambda x: 1 if x == 1 else 0)
            
            df_for_task_weights = df_processed[df_processed['branch'].notna() & (df_processed['branch'] != '')]
            
            if not df_for_task_weights.empty:
                task_quality = df_for_task_weights.groupby("branch")["is_quality_binary"].mean()
                for task, quality in task_quality.items():
                    new_rules["task_weights"][task] = min(1.5, max(0.7, quality * 2))
                logger.info("更新了任务权重。")
            else:
                logger.warning("过滤后 'branch' 列为空，无法更新任务权重。")
        else:
            logger.warning("无法根据 'branch' 或 'is_quality' 列更新任务权重，可能为空或不存在。")

        return new_rules
    except Exception as e:
        logger.error(f"规则生成失败: {e}", exc_info=True)
        return rule_engine.current_rules.copy()

def save_to_finetune(df: pd.DataFrame, output_path: str):
    try:
        base_cols_to_save = ["prompt", "result_info", "file_url", "branch", "is_quality", "stay_time", "score", "has_image_data"]
        image_feature_cols = []
        if _clip_model is not None and _clip_projection_dim is not None:
            image_feature_cols = [f"image_feature_{i}" for i in range(_clip_projection_dim)]

        all_cols_to_save = list(set(base_cols_to_save + image_feature_cols))
        actual_cols = [col for col in all_cols_to_save if col in df.columns and col not in ['image_raw_data_list', 'image_object_keys_list']]

        df[actual_cols].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条高质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存高质量数据失败: {e}", exc_info=True)
        raise

def save_to_rl(df: pd.DataFrame, output_path: str):
    try:
        base_cols_to_save = ["prompt", "result_info", "file_url", "branch", "is_quality", "stay_time", "score", "has_image_data"]
        image_feature_cols = []
        if _clip_model is not None and _clip_projection_dim is not None:
            image_feature_cols = [f"image_feature_{i}" for i in range(_clip_projection_dim)]

        all_cols_to_save = list(set(base_cols_to_save + image_feature_cols))
        actual_cols = [col for col in all_cols_to_save if col in df.columns and col not in ['image_raw_data_list', 'image_object_keys_list']]

        df[actual_cols].to_csv(output_path, index=False)
        logger.info(f"已保存 {len(df)} 条中质量数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存中质量数据失败: {e}", exc_info=True)
        raise