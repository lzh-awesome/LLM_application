"""Prompt 模板模块 - 集中管理所有 Prompt"""

from typing import List


class PromptTemplates:
    """集中管理所有 Prompt 模板"""

    # ==================== 通用问答模板 ====================

    QA_PROMPT_TEMPLATE = """
请优先根据问答对资料回答用户的问题，如果问答对资料对你没有帮助，你可以试着通过自身的知识来回答，如果还是无法回答，你可以说"无法回答该问题，我需要更多的信息。"

问答对资料：
{qa_text}

用户问题："{question}"
"""

    # ==================== 简洁/深入模式模板 ====================

    CONCISE_PROMPT_TEMPLATE = """
你是一个专业的问答助手，专注于提供简洁明了的回答。请根据以下对话历史，用最简短的语言给出准确的答案，避免任何不必要的解释或背景信息。

回答格式要求：
- 直接回答问题的核心内容，答案要是中文的。
- 如果问题涉及复杂背景，仅提供关键结论。
- 如果问"你是谁？"，你需要回答你是**小智**，你的智能AI助手。
- 如果你的答案中包含人物、机构、装备、基地、设施类的实体，请把这些实体的内容用**加粗，比如**基思·亚历山大**、**美国海军部**、**B-2幽灵式战略轰炸机**、**诺福克海军基地**、**朴茨茅斯海军造船厂**。
- 如果问"你使用的大语言模型推理框架是什么?",你需要回答我采用的是LangChain-Chatchat推理框架，可以支持多种大语言模型接入。
"""

    DETAILED_PROMPT_TEMPLATE = """
你是一个专业的问答助手，专注于提供深入、详细的回答。请根据以下对话历史，尽可能全面地解答，包括背景知识、相关细节和扩展信息。如果问题涉及多个方面，请分点说明。

回答格式要求：
- 提供完整的背景信息和上下文。
- 使用清晰的逻辑结构，必要时分点说明。
- 包括相关的扩展内容或建议。
- 回答应至少包含三段内容，答案要是中文的。
- 如果问"你是谁？"，你需要回答你是**小智**，你的智能AI助手。
- 如果你的答案中包含人物、机构、装备、基地、设施类的实体，请把这些实体的内容用**加粗，**基思·亚历山大**、**美国海军部**、**B-2幽灵式战略轰炸机**、**诺福克海军基地**、**朴茨茅斯海军造船厂**。
- 如果问"你使用的大语言模型推理框架是什么?",你需要回答我采用的是LangChain-Chatchat推理框架，可以支持多种大语言模型接入。
"""

    # ==================== 知识图谱问答模板 ====================

    ENTITY_ISOLATED_PROMPT = """
你是一个知识图谱问答系统。专注于提供深入、详细的回答。请根据以下对话历史，尽可能全面地解答，包括背景知识、相关细节和扩展信息。如果问题涉及多个方面，请分点说明。
用户问题是："{question}"
以下是你得到的实体信息：
{entity_desc}
{accumulated_info}
这是提供你给你参考的实体信息，请根据该信息来尝试回答问题。
如果该信息对你没有帮助，你可以试着通过自身的知识来回答，如果还是无法回答，你可以说"无法回答该问题，我需要更多的信息。"
如果你的答案中包含人物、机构、装备、基地、设施类的实体，请把这些实体的内容用**加粗，比如**基思·亚历山大**、**美国海军部**、**B-2幽灵式战略轰炸机**、**诺福克海军基地**、**朴茨茅斯海军造船厂**。
"""

    MULTI_HOP_JUDGMENT_PROMPT = """
用户问题是："{question}"

给定以下信息：
1. 当前跳数的最相关三元组：{best_triple}
2. 当前跳数的相关其他三元组：{other_three_triples}
3. 头实体属性信息：{from_entity_desc}
4. 尾实体属性信息：{to_entity_desc}
5. 问答对信息：{qa_text}

累积的多跳查询信息：
-{accumulated_info}

请根据上述所有信息判断：
- 当前所有信息是否足以回答用户的问题？
- 如果可以，请直接根据当前所有信息回答用户的问题，优先使用头实体属性信息，然后参考三元组的信息；
- 如果不可以，请回答"当前信息不足，无法回答该问题，需要进一步查询"，并解释为什么这些信息不足以回答用户问题。

注意：
    三元组信息与头实体属性信息、尾实体的属性信息互为补充，并不冲突；
    请综合考虑所有跳数的信息，而不仅仅是当前跳数的信息。
    请尽可能全面地解答，包括背景知识、相关细节和扩展信息。如果问题涉及多个方面，请分点说明
"""

    # ==================== 研究报告模板 ====================

    RESEARCH_REPORT_PROMPT = """
# 任务要求
请根据以下要求生成一份专业的研究报告：

# 用户需求
{user_query}

# 格式要求
1. **报告结构**：
   - 标题：简明扼要，突出核心主题
   - 摘要：200字以内，概括报告核心观点
   - 正文：分章节展开，每章有明确标题
   - 结论：总结核心发现和建议
   - 参考资料：列出信息来源

2. **内容要求**：
   - 信息准确，数据可靠，引用权威来源
   - 分析深入，观点清晰，逻辑严密
   - 语言专业，表述客观，避免主观臆断
   - 适当使用图表、数据可视化辅助说明

3. **风格要求**：
   - 学术风格，正式严谨
   - 使用专业术语，但避免晦涩难懂
   - 段落分明，层次清晰

4. **时间信息**：
   报告生成日期：{current_date}

请按照上述要求，生成一份高质量的研究报告。
"""

    OPERATIONAL_PLAN_PROMPT = """
# 任务要求
请根据以下要求生成一份专业的作战规划：

# 用户需求
{user_query}

# 格式要求
1. **规划结构**：
   - 行动名称：简洁有力，体现行动性质
   - 战略背景：简述行动的战略意义和必要性
   - 目标分析：明确行动目标和预期效果
   - 方案设计：详细的作战方案和实施步骤
   - 资源配置：人员、装备、后勤等资源分配
   - 风险评估：潜在风险和应对措施
   - 总结建议：关键要点和后续建议

2. **内容要求**：
   - 战略定位准确，目标明确可行
   - 方案设计合理，步骤详细具体
   - 资源配置科学，风险评估全面
   - 语言专业规范，表述客观严谨

3. **时间信息**：
   规划生成日期：{current_date}

请按照上述要求，生成一份高质量的作战规划。
"""

    # ==================== 工具调用系统提示 ====================

    TOOL_CALLING_SYSTEM_PROMPT = "你是一个工具调用助手，根据用户最新请求判断是否需要调用工具。"


class PromptBuilder:
    """Prompt 构建器"""

    @staticmethod
    def build_qa_prompt(qa_text: str, question: str) -> str:
        """构建问答 Prompt"""
        return PromptTemplates.QA_PROMPT_TEMPLATE.format(
            qa_text=qa_text,
            question=question,
        )

    @staticmethod
    def build_entity_isolated_prompt(
        question: str,
        entity_desc: str,
        accumulated_info: str = "",
    ) -> str:
        """构建孤立实体 Prompt"""
        return PromptTemplates.ENTITY_ISOLATED_PROMPT.format(
            question=question,
            entity_desc=entity_desc,
            accumulated_info=accumulated_info,
        )

    @staticmethod
    def build_multi_hop_judgment_prompt(
        question: str,
        best_triple: List,
        other_three_triples: List,
        from_entity_desc: str,
        to_entity_desc: str,
        qa_text: str,
        accumulated_info: str,
    ) -> str:
        """构建多跳判断 Prompt"""
        return PromptTemplates.MULTI_HOP_JUDGMENT_PROMPT.format(
            question=question,
            best_triple=best_triple,
            other_three_triples=other_three_triples,
            from_entity_desc=from_entity_desc,
            to_entity_desc=to_entity_desc,
            qa_text=qa_text,
            accumulated_info=accumulated_info,
        )

    @staticmethod
    def build_research_report_prompt(user_query: str, current_date: str) -> str:
        """构建研究报告 Prompt"""
        return PromptTemplates.RESEARCH_REPORT_PROMPT.format(
            user_query=user_query,
            current_date=current_date,
        )

    @staticmethod
    def build_operational_plan_prompt(user_query: str, current_date: str) -> str:
        """构建作战规划 Prompt"""
        return PromptTemplates.OPERATIONAL_PLAN_PROMPT.format(
            user_query=user_query,
            current_date=current_date,
        )