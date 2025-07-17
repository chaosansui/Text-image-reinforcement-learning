import pandas as pd
import logging
from typing import Optional, List, Dict, Any

# 导入配置
from config import DB_CONFIG, MINIO_CONFIG, OUTPUT_FILES

# 导入工具函数和质量逻辑
from utils.db_utils import fetch_data_from_db
from utils.minio_utils import initialize_minio_client, get_minio_client, read_image_from_minio, list_objects_by_prefix
from quality_logic import RuleEngine, process_data, split_data, save_to_finetune, save_to_rl

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_pipeline():
    # 1. 初始化 MinIO 客户端
    logger.info("正在初始化 MinIO 客户端...")
    initialize_minio_client(MINIO_CONFIG)
    minio_client = get_minio_client()

    # 初始化规则引擎
    rule_engine = RuleEngine()

    try:
        # 2. 从数据库读取数据
        logger.info("开始从数据库获取数据...")
        df = fetch_data_from_db(**DB_CONFIG)

        if df.empty:
            logger.warning("获取到的数据为空，请检查数据库连接和查询条件。")
            return

        # --- NEW FILTERING LOGIC ---
        # Define supported file extensions
        supported_extensions = ('.doc', '.docx', '.pdf', '.jpg', '.png', '.jpeg')

        # Filter out rows where file_url is not valid or has an unsupported extension
        original_row_count = len(df)
        
        # Ensure 'file_url' is string type and not NaN/empty
        df_filtered = df[
            df['file_url'].notna() &
            (df['file_url'].astype(str).str.strip() != '') &
            df['file_url'].astype(str).str.lower().str.endswith(supported_extensions)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if len(df_filtered) < original_row_count:
            logger.info(f"已过滤掉 {original_row_count - len(df_filtered)} 条具有不支持文件格式的记录。")
            logger.info(f"剩余 {len(df_filtered)} 条记录用于处理。")

        if df_filtered.empty:
            logger.warning("过滤后数据为空，没有符合条件的文件URL进行处理。")
            return
        
        df = df_filtered # Use the filtered DataFrame for subsequent steps
        # --- END NEW FILTERING LOGIC ---


        # 为存储图片数据和读取状态的列表
        all_image_raw_data_list = []
        all_image_read_success_list = []
        all_image_object_keys_list = [] # 存储下载的图片对应的对象键列表

        # 3. 从 MinIO 读取图片数据 (如果 MinIO 客户端已成功初始化)
        if minio_client and "file_url" in df.columns:
            logger.info(f"尝试从 MinIO 存储桶 '{MINIO_CONFIG['bucket_name']}' 读取文件。")
            minio_bucket_name = MINIO_CONFIG['bucket_name']

            for index, row in df.iterrows():
                file_url = row['file_url']
                current_record_images = []
                current_record_success = False
                current_record_object_keys = []

                # No need for pd.notna and strip checks here, as they're handled by the initial filter
                file_url_lower = file_url.lower()

                if file_url_lower.endswith(('.doc', '.docx', '.pdf')):
                    base_path_without_suffix = file_url.rsplit('.', 1)[0]
                    image_prefix = base_path_without_suffix 
                    logger.debug(f"处理文档类型 file_url: {file_url}, 提取图片前缀: {image_prefix}")

                    potential_image_objects = list_objects_by_prefix(
                        minio_client, minio_bucket_name, prefix=image_prefix, recursive=True
                    )
                    
                    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
                    png_objects = [obj for obj in potential_image_objects if obj.lower().endswith(image_extensions)]

                    if png_objects:
                        for obj_key in png_objects:
                            image_bytes = read_image_from_minio(minio_client, minio_bucket_name, obj_key)
                            if image_bytes:
                                current_record_images.append(image_bytes)
                                current_record_object_keys.append(obj_key)
                                current_record_success = True
                            else:
                                logger.warning(f"未能读取图片: {obj_key}")
                    else:
                        logger.debug(f"在前缀 '{image_prefix}' 下未找到任何支持的图片。")

                elif file_url_lower.endswith(('.jpg', '.png', '.jpeg')):
                    full_image_path = f"Temp/{file_url}"
                    logger.debug(f"处理图片类型 file_url: {file_url}, 尝试直接读取路径: {full_image_path}")
                    
                    image_bytes = read_image_from_minio(minio_client, minio_bucket_name, full_image_path)
                    if image_bytes:
                        current_record_images.append(image_bytes)
                        current_record_object_keys.append(full_image_path)
                        current_record_success = True
                    else:
                        logger.warning(f"未能直接读取图片: {full_image_path}")
                # No 'else' block needed here for unsupported formats, as they were already filtered out

                final_image_list = current_record_images if current_record_images else None
                final_success_status = bool(final_image_list) # True only if final_image_list is not None and not empty

                all_image_raw_data_list.append(final_image_list)
                all_image_read_success_list.append(final_success_status)
                all_image_object_keys_list.append(current_record_object_keys if current_record_object_keys else None)

            df['image_raw_data_list'] = all_image_raw_data_list
            df['image_read_success'] = all_image_read_success_list
            df['image_object_keys_list'] = all_image_object_keys_list

            logger.info(f"总计成功读取图片数据的记录数: {df['image_read_success'].sum()} 条。")
        else:
            logger.warning("MinIO 客户端未初始化或 DataFrame 中没有 'file_url' 列，跳过从 MinIO 读取文件。")
            df['image_raw_data_list'] = None
            df['image_read_success'] = False
            df['image_object_keys_list'] = None
            if "file_url" not in df.columns:
                df["file_url"] = None

        # 4. 处理数据并生成新规则
        logger.info("开始处理数据并更新规则...")
        new_rules = process_data(df, rule_engine)
        rule_engine.update_rules(new_rules)

        # 5. 分流数据
        logger.info("开始数据分流...")
        high_quality, mid_quality, low_quality = split_data(df, rule_engine)

        # 6. 保存结果
        logger.info("保存结果文件...")
        cols_to_save = ["prompt", "result_info", "file_url", "branch", "is_quality", "stay_time", "score", "has_image_data", "image_read_success", "image_object_keys_list"]
        actual_cols = [col for col in cols_to_save if col in df.columns and col not in ['image_raw_data_list']]
        
        if not high_quality.empty:
            high_quality[actual_cols].to_csv(OUTPUT_FILES["high_quality"], index=False)
            logger.info(f"已保存 {len(high_quality)} 条高质量数据到 {OUTPUT_FILES['high_quality']}")
        if not mid_quality.empty:
            mid_quality[actual_cols].to_csv(OUTPUT_FILES["mid_quality"], index=False)
            logger.info(f"已保存 {len(mid_quality)} 条中质量数据到 {OUTPUT_FILES['mid_quality']}")
        # Low quality samples are not saved to a specific file in your original code, but they are tracked in stats.

        # 7. 输出统计信息
        logger.info("\n" + "="*50)
        logger.info(f"数据分流完成 - 总计: {len(df)} 条")
        logger.info(f"高质量样本: {len(high_quality)} 条 ({len(high_quality)/len(df):.1%})")
        logger.info(f"中质量样本: {len(mid_quality)} 条 ({len(mid_quality)/len(df):.1%})")
        logger.info(f"低质量样本: {len(low_quality)} 条 ({len(low_quality)/len(df):.1%})")

        if not high_quality.empty:
            sample = high_quality.iloc[0]
            logger.info("\n高质量样本示例:")
            logger.info(f"Prompt: {sample['prompt'][:100]}...")
            logger.info(f"Result: {str(sample['result_info'])[:100]}...")
            logger.info(f"Score: {sample.get('score', 'N/A')}")
            if 'image_read_success' in sample:
                logger.info(f"图片读取成功: {sample['image_read_success']}")
                if sample['image_read_success'] and sample['image_raw_data_list'] is not None:
                    logger.info(f"读取到 {len(sample['image_raw_data_list'])} 张图片。")
                    if sample['image_raw_data_list']:
                         logger.info(f"第一张图片数据大小: {len(sample['image_raw_data_list'][0])} 字节")
                         logger.info(f"第一张图片对象键: {sample['image_object_keys_list'][0]}")

        logger.info("\nCurrent Rule Parameters:")
        for k, v in rule_engine.current_rules.items():
            logger.info(f"{k}: {v}")

    except Exception as e:
        logger.error(f"主流程执行失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_data_pipeline()