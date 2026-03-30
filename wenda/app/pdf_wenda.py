import os
import json
import numpy as np
import re
import asyncio
import time
import uuid
import torch
import logging
import argparse
import yaml  # <<< 新增: 导入yaml库
import hashlib  # <<< 在文件顶部已导入，无需重复

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber
import aiohttp
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

# 导入 pynvml 用于 GPU 显存监控
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, \
        NVMLError_DriverNotLoaded

    _PYNVML_AVAILABLE = True
except (ImportError, NVMLError_DriverNotLoaded):
    _PYNVML_AVAILABLE = False
    print(
        "Warning: pynvml not installed or NVIDIA driver not loaded. GPU memory monitoring will not be available. Install with 'pip install pynvml' for optimal GPU selection.")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
OLLAMA_API_URL = "http://10.117.1.238:7003/api/generate"

# --- 全局变量和启动逻辑 ---
# 存储 SentenceTransformer 模型实例，确保只加载一次
sentence_transformer_model: Optional[SentenceTransformer] = None
# 存储 SentenceTransformer 模型的名称
EMBEDDING_MODEL_NAME = "/workspace/codes/deepseek/deepseek_model/bge-m3"

# 全局存储，用于管理不同会话的RAG系统实例
rag_systems: Dict[str, 'RAGSystem'] = {}


def find_best_gpu_device() -> Optional[str]:
    """
    查找当前显存最空闲的GPU设备。
    """
    if not _PYNVML_AVAILABLE:
        logger.warning("pynvml未安装或驱动未加载，无法检测GPU显存，将尝试使用默认CUDA设备。")
        # 尝试返回 'cuda'，让 PyTorch 自动选择
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        nvmlInit()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.info("未检测到CUDA设备，将使用CPU。")
            return 'cpu'

        best_device = None
        max_free_memory = -1

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            free_memory_mib = info.free // (1024 * 1024)  # 转换为MiB
            logger.info(
                f"GPU {i}: 总显存 {info.total / (1024 * 1024 * 1024):.2f} GiB, 已用 {info.used / (1024 * 1024 * 1024):.2f} GiB, 空闲 {free_memory_mib} MiB")

            if free_memory_mib > max_free_memory:
                max_free_memory = free_memory_mib
                best_device = f'cuda:{i}'

        logger.info(f"选择显存最空闲的设备: {best_device} (空闲显存: {max_free_memory} MiB)")
        return best_device

    except Exception as e:
        logger.error(f"查找最佳GPU设备时发生错误: {e}")
        # 发生错误时，尝试返回 'cuda'，让 PyTorch 自动选择
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    finally:
        if _PYNVML_AVAILABLE:
            nvmlShutdown()


def remove_ref_tags(text):
    # 正则表达式匹配 [[REF:第数字页片段数字]]
    pattern = r'\[\[REF:第\d+页片段\d+\]\]'
    # 使用 re.sub 替换为空字符串（即删除）
    cleaned_text = re.sub(pattern, '', text)
    # 去除多余的空格和换行符
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


@app.on_event("startup")
async def load_embedding_model_on_startup():
    """
    FastAPI 启动时加载 SentenceTransformer 模型。
    """
    global sentence_transformer_model
    if sentence_transformer_model is None:
        try:
            import torch  # 确保 torch 已导入，用于检查 cuda 可用性
            device = find_best_gpu_device()
            if device is None:  # Fallback if find_best_gpu_device fails or no GPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.warning(f"无法确定最佳GPU设备，将尝试使用: {device}")

            logger.info(f"FastAPI 启动中：加载 SentenceTransformer 模型 '{EMBEDDING_MODEL_NAME}' 到设备 '{device}'...")
            sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            logger.info("SentenceTransformer 模型加载完成。")
        except Exception as e:
            logger.error(f"SentenceTransformer 模型加载失败: {e}")
            # 如果模型加载失败，考虑是否需要停止服务或提供降级方案
            raise RuntimeError(f"模型加载失败，无法启动应用: {e}")


@dataclass
class DocumentChunk:
    """文档块数据结构"""
    text: str
    page_num: int
    chunk_id: str
    pdf_filename: str  # 新增：PDF的原始文件名
    embedding: Optional[np.ndarray] = None


class PDFProcessor:
    """PDF文档处理器"""

    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_file: bytes) -> List[Dict[str, Any]]:
        """从PDF文件中提取文本"""
        pages_text = []
        try:
            with BytesIO(pdf_file) as pdf_file_path:
                with pdfplumber.open(pdf_file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():  # 只保存非空页面
                            pages_text.append({
                                'page_num': page_num,
                                'text': text.strip()
                            })
            logger.info(f"使用pdfplumber成功提取文本，共{len(pages_text)}页")
        except Exception as e:
            logger.error(f"PDF读取错误: {e}")
            raise
        return pages_text

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空格，包括换行符等
        text = re.sub(r'\s+', ' ', text)
        # 过滤掉一些特殊字符，保留中文、英文、数字和常用标点
        # 这里可以根据实际需求调整，例如保留逗号、句号等
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,?!;]', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """将文本分块"""
        # 使用更灵活的分隔符，包括句号、问号、感叹号、分号、换行符等
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:  # +1 for potential space
                chunks.append(current_chunk.strip())
                # 确保 overlap 不会超过当前块的长度，避免负索引
                current_chunk_len = len(current_chunk)
                if overlap > 0 and current_chunk_len > overlap:
                    current_chunk = current_chunk[current_chunk_len - overlap:].strip() + " " + sentence
                else:
                    current_chunk = sentence  # 如果 overlap 不适用或太小，直接开始新块
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)  # Add space if not first sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def process_pdf(self, pdf_file: bytes, pdf_filename: str, chunk_size: int = 500, overlap: int = 50) -> List[
        DocumentChunk]:
        """处理PDF文件，返回文档块列表"""
        logger.info(f"开始处理PDF文件: {pdf_filename} ")
        pages_data = self.extract_text_from_pdf(pdf_file)
        chunks = []
        chunk_counter = 0
        for page_data in pages_data:
            page_num = page_data['page_num']
            text = self.clean_text(page_data['text'])
            text_chunks = self.chunk_text(text, chunk_size, overlap)
            for chunk_text in text_chunks:
                if len(chunk_text.strip()) > 20:  # 过滤太短的块，避免无意义的块
                    chunk = DocumentChunk(
                        text=chunk_text,
                        page_num=page_num,
                        # 为每个chunk生成一个唯一的ID，包括页面和chunk_counter
                        # chunk_id=f"page_{page_num}_chunk_{chunk_counter}",
                        chunk_id=f"第{page_num}页片段{chunk_counter}",

                        pdf_filename=pdf_filename,
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
        logger.info(f"PDF处理完成，共生成 {len(chunks)} 个文档块")
        return chunks


class VectorStore:
    """向量存储和检索系统"""

    def __init__(self, model: SentenceTransformer):
        logger.info("初始化 VectorStore，使用预加载的 SentenceTransformer 模型。")
        self.model = model
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[DocumentChunk] = []  # 存储DocumentChunk对象
        # 添加一个字典，用于通过 chunk_id 快速查找 DocumentChunk
        self.chunk_id_map: Dict[str, DocumentChunk] = {}

    def add_documents(self, chunks: List[DocumentChunk]):
        """添加文档到向量存储"""
        logger.info(f"开始向量化 {len(chunks)} 个文档块...")
        if not chunks:
            logger.warning("没有文档块可供添加。")
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self.chunks.append(chunk)
            self.chunk_id_map[chunk.chunk_id] = chunk  # 更新 chunk_id_map

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"FAISS 索引已初始化，维度: {self.dimension}")

        self.index.add(embeddings.astype('float32'))
        logger.info(
            f"成功添加 {len(chunks)} 个文档块到向量存储, 当前总文档块数量: {len(self.chunks)}, FAISS索引大小: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if self.index is None or len(self.chunks) == 0:
            logger.warning("向量存储为空，无法进行搜索。")
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True).astype('float32')
        actual_top_k = min(top_k, self.index.ntotal)
        if actual_top_k == 0:
            return []

        D, I = self.index.search(query_embedding, actual_top_k)
        scores = D[0]
        indices = I[0]

        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk.text,
                    'page_num': chunk.page_num,
                    'chunk_id': chunk.chunk_id,  # 确保这里包含了 chunk_id
                    'pdf_filename': chunk.pdf_filename,
                    'score': float(score),
                    'rank': i + 1
                })
        logger.info(f"查询 '{query}' 找到 {len(results)} 个相关文档。")
        return results

    def save_index(self, save_path: str):
        """保存索引和文档"""
        os.makedirs(save_path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'text': chunk.text,
                'page_num': chunk.page_num,
                'chunk_id': chunk.chunk_id,
                'pdf_filename': chunk.pdf_filename,  # 包含 pdf_filename
                'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
            })
        with open(os.path.join(save_path, "chunks.json"), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"索引已保存到: {save_path}")

    def load_index(self, load_path: str):
        """加载索引和文档"""
        index_path = os.path.join(load_path, "faiss_index.bin")
        chunks_path = os.path.join(load_path, "chunks.json")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS索引已从 {index_path} 加载")

            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.chunks = []
            self.chunk_id_map = {}  # 清空并重新构建
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    text=chunk_data['text'],
                    page_num=chunk_data['page_num'],
                    chunk_id=chunk_data['chunk_id'],
                    pdf_filename=chunk_data['pdf_filename'],
                    embedding=np.array(chunk_data['embedding'], dtype=np.float32) if chunk_data['embedding'] else None
                )
                self.chunks.append(chunk)
                self.chunk_id_map[chunk.chunk_id] = chunk  # 重新填充 chunk_id_map
            logger.info(f"文档块已从 {chunks_path} 加载，共 {len(self.chunks)} 个。")
            return True
        else:
            self.index = None
            self.chunks = []
            self.chunk_id_map = {}
            logger.warning(f"未找到会话数据文件: {load_path}。FAISS索引或chunks文件缺失。")
            return False


# --- 新增: 配置加载函数 ---
def load_config():
    """
    从 YAML 文件加载配置，支持通过命令行参数指定配置文件。
    """
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(description="启动问答服务")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='指定配置文件路径 (例如: --config config_prod.yaml)'
    )
    args = parser.parse_args()  # 解析命令行参数

    config_path = args.config  # 使用命令行参数指定的路径，或使用默认值 'config.yaml'

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError("配置文件为空")
            print(f"成功加载配置文件: {config_path}")  # 打印信息，方便确认
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析配置文件失败: {e}")
    except Exception as e:
        raise Exception(f"加载配置时发生未知错误: {e}")


# --- 加载配置 ---
# 使用全局变量 config 来存储加载的配置
config = load_config()  # <<< 调用函数加载配置

client_openai = AsyncOpenAI(
    base_url=config["llm"]["base_url"],
    api_key=config["llm"]["api_key"],
)

# --- 全局配置变量 (替代原来的 Config 类) ---
LLM_MODEL = config["llm"]["model"]


# 调用 OpenAI 接口获得流式响应
async def call_stream_response_by_asyncOpenAI(
        messages: List[Dict[str, str]],
        model_name: str = LLM_MODEL,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 4096
):
    """
    使用 OpenAI 兼容客户端调用 LLM 进行流式聊天补全。
    """
    try:
        response = await client_openai.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,

        )  # response是一个异步流对象，可以被async for遍历
        async for chunk in response:  # 从异步流对象中逐个获取数据块（chunk）
            if chunk.choices and chunk.choices[
                0].delta.content is not None:  # 这个 chunk 并不是一个原始的 JSON 字符串，也不是一个普通的 dict，而是一个 Pydantic 模型对象 （或类似的结构化对象），表示一个 ChatCompletionChunk;chunk 是 openai.types.chat.ChatCompletionChunk 类的实例
                yield chunk.choices[0].delta.content  # 把每个小片段返回给客户端
    except Exception as e:
        print(f"LLM Stream Error: {e}")
        yield {"error": str(e)}


# 同一个 session_id 中，若已上传过 同名且内容相同 的 PDF，则跳过处理。
class RAGSystem:
    """RAG问答系统"""

    def __init__(self, model: SentenceTransformer):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore(model)
        self.conversation_history = []
        # 新增：记录已上传的文件（filename -> md5_hash）
        self.uploaded_files: Dict[str, str] = {}

    def _compute_file_hash(self, pdf_bytes: bytes) -> str:
        """计算PDF内容的MD5哈希"""
        return hashlib.md5(pdf_bytes).hexdigest()

    def upload_and_process_pdf(self, pdf_file: bytes, pdf_filename: str, chunk_size: int = 500, overlap: int = 50):
        """上传并处理PDF文件，支持防重复"""
        file_hash = self._compute_file_hash(pdf_file)

        # 检查是否已上传过同名且同内容的文件
        if pdf_filename in self.uploaded_files:
            if self.uploaded_files[pdf_filename] == file_hash:
                logger.info(f"文件 '{pdf_filename}' 已存在且内容相同，跳过重复上传。")
                # 返回 0 表示未新增 chunk
                return 0

        # 否则正常处理
        chunks = self.pdf_processor.process_pdf(pdf_file, pdf_filename, chunk_size, overlap)
        self.vector_store.add_documents(chunks)
        self.uploaded_files[pdf_filename] = file_hash  # 记录
        return len(chunks)

    async def ask_question(self, question: str, top_k: int = 5):
        """回答问题"""
        search_results = self.vector_store.search(question, top_k)  # 列表里面存了字典
        # 用于存储检索到的文档块信息，以chunk_id为键，方便后续查找
        relevant_chunks_list = []

        if not search_results:
            context = ""
            logger.warning("未找到相关文档，将尝试在没有上下文的情况下生成回答。")
        else:
            context_texts = []
            for result in search_results:
                # 格式化上下文，加入 SOURCE 标记，指示大模型可以引用的 chunk_id
                context_texts.append(f"[[REF:{result['chunk_id']}]] {result['text']}")

                # 存储检索到的原始信息，用于在LLM引用时查找
                new_chunk = {
                    'pdf_filename': result['pdf_filename'],
                    'chunk_id': result['chunk_id'],
                    'text': result['text'],
                    'score': float(result['score']),
                }
                relevant_chunks_list.append(new_chunk)

            context = "\n\n".join(context_texts)

        history_str = ""
        if self.conversation_history:
            history_str = "\n".join([f"用户: {h['question']}\n助手: {h['answer']}" for h in self.conversation_history])
            history_str = f"以下是之前的对话历史：\n{history_str}\n"

        # 构造 OpenAI 兼容的 messages 格式
        messages_for_llm = [
            {"role": "system", "content": (
                "你是一个知识渊博的助手。请基于提供的文档内容和对话历史来回答用户的问题。\n"
                "当你的回答引用了文档中的具体内容时，请在引用句的末尾使用 `[[REF:chunk_id]]` 的格式进行标注，其中 `chunk_id` 是文档内容前的 `[[REF:chunk_id]]` 中的唯一标识符。\n"
                "例如：'根据文档，全球变暖是一个严峻的问题。[[REF:第1页片段1]]'\n"
                "请确保只引用你实际使用的文档片段的 `chunk_id`。\n"
                "如果文档内容中没有提及相关信息，请结合你自身的知识库来尝试回答问题，此时无需添加引用。\n"
                "如果实在回答不了这个问题，请表示你自身无法回答这个问题。\n\n"
                f"{history_str.strip()}"
            )},
            {"role": "user", "content": f"文档内容：\n{context}\n\n用户问题：\n{question}"}
        ]  # 这里把 history_str 放进了 system message，因为它是对模型行为的约束；而 context + question 作为 user 输入，符合 RAG 的典型模式

        async def stream_response():
            full_response = ""
            try:
                async for new_text in call_stream_response_by_asyncOpenAI(messages_for_llm):

                    if isinstance(new_text, dict) and "error" in new_text:
                        yield f"data: {json.dumps(new_text, ensure_ascii=False)}\n\n"
                        return

                    full_response += new_text

                    chunk = {
                        "id": "llm_thinking_process",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": LLM_MODEL,
                        "choices": [
                            {
                                "delta": {"content": new_text},
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": None,
                            }
                        ],
                        "service_tier": None,
                        "system_fingerprint": None
                    }
                    json_chunk = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {json_chunk}\n\n"
                    await asyncio.sleep(0.001)

                source_chunk = {
                    "id": "llm_thinking_process",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "delta": {"content": "\n\n\n###### 资料来源"
                                                 "\n\n  大模型 "},
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ],
                    "service_tier": None,
                    "system_fingerprint": None
                }
                json_source_chunk = json.dumps(source_chunk, ensure_ascii=False)
                yield f"data: {json_source_chunk}\n\n"

                # LLM流式输出结束后，处理完整的回答和提取溯源信息
                final_llm_response = extract_content_after_think(full_response)

                # 提取引用的 chunk_id 并清理文本
                cleaned_answer = remove_ref_tags(final_llm_response)

                print("\n最终回答文本 (未清除引用标记):\n", final_llm_response)
                print("\n最终回答文本 (已清除引用标记):\n", cleaned_answer)
                print("\n提取到的引用来源:")
                for relevant_chunk in relevant_chunks_list:
                    print(relevant_chunk)
                    print("\n")

                print("\n提取到的引用来源长度:\n", len(relevant_chunks_list))

                # 将完整的回答添加到对话历史，并限制最多保留最近 4 轮
                new_turn = {"question": question, "answer": cleaned_answer}
                self.conversation_history.append(new_turn)

                # 只保留最近 4 轮对话
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]
                logger.info(f"对话历史已更新，当前历史长度：{len(self.conversation_history)}")

                # 发送最终结果，包含完整回答和来源
                final_result = {
                    "question": question,
                    "answer": cleaned_answer,
                    "sources": relevant_chunks_list  # 这里将包含溯源信息
                }
                json_final_result = json.dumps(final_result, ensure_ascii=False)
                yield f"data: {json_final_result}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"流式生成回答时发生错误: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': f'流式生成回答时发生错误: {str(e)}'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    def save_system(self, session_id: str, save_base_path: str = "./rag_data"):
        """保存系统状态到会话ID对应的子目录"""
        session_path = os.path.join(save_base_path, session_id)
        os.makedirs(session_path, exist_ok=True)
        self.vector_store.save_index(session_path)
        with open(os.path.join(session_path, "conversation_history.json"), 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        logger.info(f"会话 {session_id} 的数据已保存到: {session_path}")

    def load_system(self, session_id: str, load_base_path: str = "./rag_data"):
        """加载系统状态从会话ID对应的子目录"""
        session_path = os.path.join(load_base_path, session_id)
        if not os.path.exists(session_path):
            logger.warning(f"会话 {session_id} 的数据目录 {session_path} 不存在。")
            return False
        if not self.vector_store.load_index(session_path):
            logger.error(f"加载会话 {session_id} 的VectorStore失败。")
            return False

        conversation_history_path = os.path.join(session_path, "conversation_history.json")
        if os.path.exists(conversation_history_path):
            try:
                with open(conversation_history_path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"会话 {session_id} 的对话历史已从 {conversation_history_path} 加载。")
            except json.JSONDecodeError as e:
                logger.error(f"加载会话 {session_id} 的对话历史失败 (JSON解析错误): {e}")
                self.conversation_history = []
                return False
        else:
            self.conversation_history = []
            logger.warning(f"会话 {session_id} 的对话历史文件 {conversation_history_path} 不存在。")
        return True


# 提取 <think>\n</think>\n\n 后面的内容
def extract_content_after_think(text: str) -> str:
    # 定义分隔符
    think_start = "<think>"
    think_end = "</think>\n\n"

    # 查找分隔符的位置
    start_index = text.find(think_start)
    end_index = text.find(think_end)

    # 如果找到分隔符，则提取后面的内容
    if start_index != -1 and end_index != -1:
        # 提取 <think></think> 之后的内容
        content_after_think = text[end_index + len(think_end):].strip()
        return content_after_think

    # 如果没有找到分隔符，返回原始文本
    return text.strip()


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/upload_pdf")
async def upload_pdf(
        pdf_file: UploadFile = File(...),
        session_id: Optional[str] = Form(None)
):
    global sentence_transformer_model
    if sentence_transformer_model is None:
        raise HTTPException(status_code=500, detail="SentenceTransformer 模型尚未加载。服务可能未正确启动。")

    pdf_filename = pdf_file.filename  # 获取原始文件名

    # 上传多pdf
    if session_id:
        logger.info(f"将PDF '{pdf_filename}' 添加到现有会话ID：{session_id}。")
        rag = rag_systems.get(session_id)
        if not rag:
            rag_new = RAGSystem(model=sentence_transformer_model)
            if rag_new.load_system(session_id):
                rag = rag_new
                rag_systems[session_id] = rag
                logger.info(f"会话ID '{session_id}' 从磁盘加载成功。")
            else:
                raise HTTPException(status_code=404,
                                    detail=f"提供的会话ID '{session_id}' 不存在或已过期。请启动一个新的会话或提供有效的ID。")
        logger.info(f"将PDF添加到现有会话ID: {session_id}")

    # 上传单pdf
    else:
        session_id = str(uuid.uuid4())
        rag = RAGSystem(model=sentence_transformer_model)
        rag_systems[session_id] = rag
        logger.info(f"创建新会话ID: {session_id}")
        logger.info(f"将PDF '{pdf_filename}' 添加到会话ID：{session_id}。")

    try:
        pdf_content = await pdf_file.read()
        chunk_count = rag.upload_and_process_pdf(pdf_content, pdf_filename, chunk_size=500, overlap=50)
        logger.info(f"✅ 成功处理PDF，生成了 {chunk_count} 个文档块，会话ID: {session_id}")

        rag.save_system(session_id)

        if chunk_count == 0:
            message = "PDF 已存在，跳过重复上传"
        else:
            message = "PDF处理成功"

        return JSONResponse(content={
            "message": message,
            "pdf_filename": pdf_filename,
            "session_id": session_id,
            "chunk_count": chunk_count,
            "total_chunks_in_session": rag.vector_store.index.ntotal if rag.vector_store.index else 0
        })


    except Exception as e:
        logger.error(f"PDF上传或处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF上传或处理失败: {str(e)}")


@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    根据session_id和问题进行多轮对话。
    """
    session_id = request.session_id
    question = request.question

    rag = rag_systems.get(session_id)

    if not rag:
        # 如果内存中没有，尝试从磁盘加载
        rag_new = RAGSystem(model=sentence_transformer_model)
        if rag_new.load_system(session_id):
            rag = rag_new
            rag_systems[session_id] = rag  # 重新放入内存
            logger.info(f"会话ID '{session_id}' 从磁盘加载成功，用于聊天。")
        else:
            raise HTTPException(status_code=404, detail=f"会话ID '{session_id}' 不存在或已过期。请先上传PDF。")

    logger.info(f"会话ID: {session_id}, 收到问题: {question}")
    return await rag.ask_question(question, top_k=3)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7010)