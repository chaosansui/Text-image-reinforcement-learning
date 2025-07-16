import pandas as pd
import logging
from quality_split import RuleEngine, fetch_data_from_db, split_data, process_data, save_to_finetune, save_to_rl

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # MySQL 连接参数
    db_config = {
        "host": "192.168.80.196",
        "port": 3306,
        "user": "chinait",
        "password": "FunkiAI_2025",
        "database": "funki_ai",
        "table_name": "kyc_verify_result_log",
        "fields": [
            "user_id", 
            "prompt", 
            "result_info", 
            "create_time", 
            "stay_time", 
            "is_quality",
            "branch"
        ]
    }

    # 输出文件配置
    output_files = {
        "high_quality": "./output_file/high_quality_samples.csv",
        "mid_quality": "./output_file/mid_quality_samples.csv",
        "low_quality": "./output_file/low_quality_samples.csv"
    }

    # 初始化规则引擎
    rule_engine = RuleEngine()
    
    try:
        # 从数据库读取数据
        logger.info("开始从数据库获取数据...")
        df = fetch_data_from_db(**db_config)
        
        if df.empty:
            logger.warning("获取到的数据为空，请检查数据库连接和查询条件")
            return

        # 处理数据并生成新规则
        logger.info("开始处理数据并更新规则...")
        new_rules = process_data(df, rule_engine)
        rule_engine.update_rules(new_rules)

        # 分流数据
        logger.info("开始数据分流...")
        high_quality, mid_quality, low_quality = split_data(df, rule_engine)

        # 保存结果
        logger.info("保存结果文件...")
        if not high_quality.empty:
            save_to_finetune(high_quality, output_files["high_quality"])
        if not mid_quality.empty:
            save_to_rl(mid_quality, output_files["mid_quality"])
        
        # 输出统计信息
        logger.info("\n" + "="*50)
        logger.info(f"数据分流完成 - 总计: {len(df)} 条")
        logger.info(f"高质量样本: {len(high_quality)} 条 ({len(high_quality)/len(df):.1%})")
        logger.info(f"中质量样本: {len(mid_quality)} 条 ({len(mid_quality)/len(df):.1%})")
        logger.info(f"低质量样本: {len(low_quality)} 条 ({len(low_quality)/len(df):.1%})")
        
        # 调试输出示例数据
        if not high_quality.empty:
            sample = high_quality.iloc[0]
            logger.info("\n高质量样本示例:")
            logger.info(f"Prompt: {sample['prompt'][:100]}...")
            logger.info(f"Result: {str(sample['result_info'])[:100]}...")
            logger.info(f"Score: {sample.get('score', 'N/A')}")
        
        logger.info("\n当前规则参数:")
        for k, v in rule_engine.current_rules.items():
            logger.info(f"{k}: {v}")

    except Exception as e:
        logger.error(f"主流程执行失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()