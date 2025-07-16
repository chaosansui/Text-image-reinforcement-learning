# minio_utils.py

from minio import Minio
from minio.error import S3Error
import logging
from typing import Optional
import io # 用于处理二进制数据流

logger = logging.getLogger(__name__)

# MinIO 客户端实例，在模块加载时初始化
# 这里的 MINIO_CONFIG 需要从 config.py 导入
_minio_client_instance = None # 私有变量，表示客户端实例
_minio_config_loaded = False # 标志，确保配置只加载一次

def initialize_minio_client(config: dict) -> None:
    """初始化全局 MinIO 客户端实例。"""
    global _minio_client_instance, _minio_config_loaded
    if _minio_config_loaded: # 避免重复初始化
        return

    try:
        _minio_client_instance = Minio(
            config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=config["secure"]
        )
        # 尝试列出桶，验证连接是否成功
        _minio_client_instance.list_buckets()
        logger.info("MinIO 客户端成功初始化并连接。")
        _minio_config_loaded = True
    except S3Error as e:
        logger.error(f"连接 MinIO 失败 (S3Error): {e}", exc_info=True)
        _minio_client_instance = None
    except Exception as e:
        logger.error(f"初始化 MinIO 客户端时发生意外错误: {e}", exc_info=True)
        _minio_client_instance = None

def get_minio_client() -> Optional[Minio]:
    """获取已初始化的 MinIO 客户端实例。"""
    return _minio_client_instance

def read_image_from_minio(minio_client: Minio, bucket_name: str, object_name: str) -> Optional[bytes]:
    """
    从 MinIO 读取指定对象（如图片）的内容并以字节形式返回。
    如果对象无法读取或发生错误，则返回 None。
    """
    if minio_client is None:
        logger.error("MinIO 客户端未初始化，无法读取对象。")
        return None

    try:
        # object_name 对应 file_url 中的路径
        response = minio_client.get_object(bucket_name, object_name)
        image_bytes = response.read()
        logger.debug(f"成功从 MinIO 读取对象: {bucket_name}/{object_name}。")
        return image_bytes
    except S3Error as e:
        if e.code == 'NoSuchKey':
            logger.warning(f"MinIO 中未找到对象: {bucket_name}/{object_name}。")
        else:
            logger.error(f"从 MinIO 读取对象失败 ({object_name}): {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"读取 MinIO 对象 ({object_name}) 时发生意外错误: {e}", exc_info=True)
        return None
    finally:
        # 确保关闭响应并释放连接
        if 'response' in locals() and response:
            response.close()
            response.release_conn()