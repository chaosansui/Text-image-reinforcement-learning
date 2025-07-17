import io
from minio import Minio
from minio.error import S3Error, InvalidResponseError
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

_minio_client_instance: Optional[Minio] = None

def initialize_minio_client(config: Dict[str, Any]) -> None:
    """初始化 MinIO 客户端实例"""
    global _minio_client_instance
    if _minio_client_instance is None:
        try:
            _minio_client_instance = Minio(
                endpoint=config["endpoint"],
                access_key=config["access_key"],
                secret_key=config["secret_key"],
                secure=config.get("secure", False)
            )
            # 尝试列出桶以验证连接
            _minio_client_instance.list_buckets()
            logger.info(f"MinIO 客户端初始化成功，连接到 {config['endpoint']}")
        except S3Error as e:
            logger.error(f"MinIO S3 错误: {e}")
            _minio_client_instance = None
        except InvalidResponseError as e:
            logger.error(f"初始化 MinIO 客户端时发生无效响应错误: {e}")
            logger.error("请检查 MinIO endpoint 端口是否正确 (通常API端口是9000，而非9001)。")
            _minio_client_instance = None
        except Exception as e:
            logger.error(f"初始化 MinIO 客户端时发生意外错误: {e}", exc_info=True)
            _minio_client_instance = None

def get_minio_client() -> Optional[Minio]:
    """获取 MinIO 客户端实例"""
    return _minio_client_instance

def read_image_from_minio(client: Minio, bucket_name: str, object_name: str) -> Optional[bytes]:
    """从 MinIO 读取指定对象的二进制数据"""
    try:
        response = client.get_object(bucket_name, object_name)
        image_bytes = response.read()
        response.close()
        response.release_conn()
        return image_bytes
    except S3Error as e:
        logger.warning(f"从桶 '{bucket_name}' 读取对象 '{object_name}' 失败: {e.code} - {e.message}")
        return None
    except Exception as e:
        logger.error(f"读取 MinIO 对象 '{object_name}' 时发生意外错误: {e}", exc_info=True)
        return None

# --- 新增函数：列举 MinIO 桶中指定前缀下的所有文件 ---
def list_objects_by_prefix(client: Minio, bucket_name: str, prefix: str, recursive: bool = True) -> List[str]:

    object_names = []
    try:
        # 添加 slash '/' 到前缀末尾，确保它被视为目录前缀
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        objects = client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
        for obj in objects:
            object_names.append(obj.object_name)
        logger.info(f"在桶 '{bucket_name}' 的前缀 '{prefix}' 下找到了 {len(object_names)} 个对象。")
    except S3Error as e:
        logger.error(f"列举 MinIO 对象时发生 S3 错误 (桶: {bucket_name}, 前缀: {prefix}): {e.code} - {e.message}")
    except Exception as e:
        logger.error(f"列举 MinIO 对象时发生意外错误 (桶: {bucket_name}, 前缀: {prefix}): {e}", exc_info=True)
    return object_names