"""工具函数模块"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_value_by_lang(
    data_dict: Dict,
    langs: List[str] = None,
) -> Optional[str]:
    """
    从字典中按语言优先级取值

    Args:
        data_dict: 语言-值映射字典
        langs: 语言优先级列表

    Returns:
        提取的值，未找到返回 None
    """
    if langs is None:
        langs = ["zh-cn", "zh", "en"]

    for lang in langs:
        if lang in data_dict:
            return data_dict[lang]

    # 如果都没找到，返回第一个可用值
    if data_dict:
        return list(data_dict.values())[0]
    return None


def process_entity_data(data: Dict) -> Dict:
    """
    处理实体数据，提取 labels、descs、claims

    Args:
        data: 原始实体数据

    Returns:
        处理后的数据字典
    """
    import json

    logger.debug(f"Processing entity data, type: {type(data)}")
    result = {}

    # 处理字符串类型的输入
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict or JSON string, got {type(data)}")

    # 获取 entity 字段
    entity = data.get("entity", {})
    if isinstance(entity, str):
        try:
            entity = json.loads(entity)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string in 'entity' field: {e}")

    if not isinstance(entity, dict):
        logger.warning(f"'entity' is not a dict, got {type(entity)}. Using empty dict.")
        entity = {}

    # 处理 labels
    labels = entity.get("labels", {})
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except Exception:
            labels = {}
    result["labels"] = extract_value_by_lang(labels)

    # 处理 descs
    descs = entity.get("descs", {})
    if isinstance(descs, str):
        try:
            descs = json.loads(descs)
        except Exception:
            descs = {}
    result["descs"] = extract_value_by_lang(descs)

    # 处理 claims
    claims = entity.get("claims", {})
    if isinstance(claims, str):
        try:
            claims = json.loads(claims)
        except Exception:
            claims = {}

    for key, claim_list in claims.items():
        if claim_list and isinstance(claim_list, list):
            try:
                value = claim_list[0].get("vl", {}).get("dv", {}).get("v")
                result[key] = value
            except (IndexError, KeyError, TypeError) as e:
                logger.debug(f"Error extracting claim '{key}': {e}")
                result[key] = None
        else:
            result[key] = None

    return result


def extract_labels_and_claims(data: Dict) -> Dict:
    """
    提取 labels 和 claims 的值

    Args:
        data: 输入数据

    Returns:
        包含 labels 和 claims 的字典
    """
    from fastapi import HTTPException

    filtered_data = {}

    if "labels" in data:
        filtered_data["labels"] = data["labels"]
    else:
        raise HTTPException(status_code=400, detail="该条数据缺失必需字段 'labels'")

    if "claims" in data:
        filtered_data["claims"] = data["claims"]
    else:
        raise HTTPException(status_code=400, detail="该条数据缺失必需字段 'claims'")

    return filtered_data


def extract_data(data: Dict) -> Dict:
    """
    提取 labels 和 claims 中的特定信息

    Args:
        data: 输入数据

    Returns:
        提取后的数据字典
    """
    result = {}

    # 提取 labels 的值（优先使用中文）
    if "labels" in data:
        labels = data["labels"]
        if "zh-cn" in labels:
            result["labels"] = labels["zh-cn"]
        elif "zh" in labels:
            result["labels"] = labels["zh"]
        elif "en" in labels:
            result["labels"] = labels["en"]

    # 提取 claims 的值
    if "claims" in data:
        claims = data["claims"]
        for key, value in claims.items():
            if value is not None:
                result[key] = value

    return result