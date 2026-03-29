# 这些是 FastAPI 的路由路由器（Router），类似于 Flask 中的蓝图（Blueprint）。
作用说明：
# 1. routers/__init__.py - 模块初始化文件
这个文件负责导出所有的路由路由器，让 main.py 可以统一导入：
chat_router - 聊天问答路由
entity_router - 实体相关路由
recommend_router - 推荐路由
topic_router - 话题路由
translate_router - 翻译路由
# 2. 各个 router 文件 - 功能路由模块
每个文件定义了一组相关的 API 接口：
chat.py → /brief_or_profound 接口（简洁/深入模式问答）
entity.py → 实体查询相关接口
recommend.py → 问题推荐接口
topic.py → 话题总结接口
translate.py → 翻译接口
# 3. 为什么要用 Router？
好处：
✅ 模块化：不同功能分开在不同的文件中
✅ 易维护：修改某个功能只需改对应的文件
✅ 可复用：router 可以在多个 app 之间共享
✅ 清晰的代码结构：每个 router 可以有自己的 prefix 和 tags