# db_utils.py

import pandas as pd
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
from typing import List

logger = logging.getLogger(__name__) # 获取与模块同名的logger，或者直接用root logger

def fetch_data_from_db(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table_name: str,
    fields: List[str],
    days: int = 30
) -> pd.DataFrame:
    """
    从数据库获取最近N天的数据。
    确保返回的DataFrame包含所有指定字段，特别是'prompt', 'result_info', 'create_time'。
    """
    try:
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

        # 确保包含必要的字段，因为这些字段在质量分流逻辑中被用到
        required_for_quality_logic = ["prompt", "result_info", "create_time", "stay_time", "is_quality", "branch"]
        for field in required_for_quality_logic:
            if field not in fields:
                fields.append(field)

        # 获取最近数据
        recent_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        # 你的 SQL 查询逻辑，确保过滤掉空值
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
        logger.info(f"成功从数据库获取 {len(df)} 条数据，时间范围: {recent_date} 至今。")
        return df
    except Exception as e:
        logger.error(f"数据库查询失败: {e}", exc_info=True)
        raise # 重新抛出异常，让上层调用者处理
    finally:
        if 'engine' in locals():
            engine.dispose()