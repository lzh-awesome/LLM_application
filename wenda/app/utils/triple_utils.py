"""三元组处理工具模块"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TripleUtils:
    """三元组处理工具"""

    @staticmethod
    def extract_entities_from_triple(triple_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        从三元组文本中提取头实体和尾实体

        格式: "XX的XX是XX"
        例如: "侯雪梅的父亲是侯廷振"

        Args:
            triple_name: 三元组文本

        Returns:
            (head_entity, tail_entity) 元组，解析失败返回 (None, None)
        """
        if not triple_name:
            return None, None

        # 匹配 "XX的XX是XX" 模式
        pattern = r"^(.+?)的(.+?)是(.+)$"
        match = re.match(pattern, triple_name)

        if match:
            head_entity = match.group(1)
            tail_entity = match.group(3)
            return head_entity, tail_entity
        else:
            logger.warning(f"无法解析三元组: {triple_name}")
            return None, None

    @staticmethod
    def format_accumulated_triples(accumulated_triples: List[Dict]) -> str:
        """
        格式化累积的三元组信息

        Args:
            accumulated_triples: 累积的三元组列表

        Returns:
            格式化后的文本
        """
        if not accumulated_triples:
            return "暂无累积信息"

        formatted_info = []
        for hop_info in accumulated_triples:
            depth = hop_info["depth"]
            best = hop_info["best_triple"]
            other_three = hop_info.get("other_three_triples", [])

            hop_text = f"""
第{depth + 1}跳查询结果：
- 最相关的单个三元组：{best}
- 其他相关的三元组：{other_three}
            """.strip()

            formatted_info.append(hop_text)

        return "\n\n".join(formatted_info)

    @staticmethod
    def convert_entity_id_to_description_id(entity_id: str, collection_prefix: str = "entity_description_large") -> str:
        """
        将 entity/xxx 格式的 ID 转换为 entity_description_large/xxx 格式

        Args:
            entity_id: 原始实体 ID，如 "entity/123456"
            collection_prefix: 集合前缀

        Returns:
            转换后的 ID
        """
        return entity_id.replace("entity/", f"{collection_prefix}/")

    @staticmethod
    def convert_description_id_to_entity_id(description_id: str) -> str:
        """
        将 entity_description_large/xxx 格式的 ID 转换为 entity/xxx 格式

        Args:
            description_id: 描述集合 ID

        Returns:
            转换后的实体 ID
        """
        return description_id.replace("entity_description_large/", "entity/")

    @staticmethod
    def extract_link_id_suffix(link_id: str) -> str:
        """
        从 link_tmp/xxx 格式中提取后缀 xxx

        Args:
            link_id: 链接 ID

        Returns:
            后缀部分
        """
        return link_id.split("/")[-1] if "/" in link_id else link_id

    @staticmethod
    def build_natural_language_triple(from_entity_name: str, relation: str, to_entity_name: str) -> str:
        """
        构建自然语言格式的三元组描述

        Args:
            from_entity_name: 头实体名称
            relation: 关系名称
            to_entity_name: 尾实体名称

        Returns:
            自然语言三元组，如 "张三的父亲是李四"
        """
        if from_entity_name and to_entity_name and relation:
            return f"{from_entity_name}的{relation}是{to_entity_name}"
        return ""


class TripleFormatter:
    """三元组格式化器"""

    @staticmethod
    def format_triple_results(triples: List[Dict]) -> str:
        """
        格式化三元组查询结果为文本

        Args:
            triples: 三元组列表

        Returns:
            格式化后的文本
        """
        if not triples:
            return "暂无相关三元组"

        formatted = []
        for triple in triples:
            triple_name = triple.get("triple_name", "")
            certainty = triple.get("certainty", 0)
            formatted.append(f"- {triple_name} (相关性: {certainty:.2f})")

        return "\n".join(formatted)

    @staticmethod
    def format_qa_results(qa_results: List[Dict]) -> str:
        """
        格式化问答对查询结果为文本

        Args:
            qa_results: 问答对列表

        Returns:
            格式化后的文本
        """
        if not qa_results:
            return "\n\n问答对资料为空，请根据其他资料进行回答。\n"

        formatted = []
        for qa in qa_results:
            query = qa.get("query", "")
            answer = qa.get("answer", "")
            formatted.append(f"问：{query}\n答：{answer}\n")

        return "".join(formatted)

    @staticmethod
    def format_entity_desc(entity_data: Dict) -> str:
        """
        格式化实体描述

        Args:
            entity_data: 实体数据

        Returns:
            实体描述文本
        """
        if not entity_data:
            return ""

        name = entity_data.get("name", "")
        desc = entity_data.get("desc", "")

        if name and desc:
            return f"{name}: {desc}"
        elif desc:
            return desc
        elif name:
            return name
        return ""