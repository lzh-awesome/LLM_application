"""
图片向量化模块
使用 BGE-VL 模型进行图片向量化
"""
import os
import base64
import hashlib
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModel


class ImageEmbedding:
    """图片向量化类"""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化图片向量化模型

        Args:
            model_path: BGE-VL 模型路径
            device: 设备 (auto, cuda, cpu)
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            print(f"正在加载图片向量化模型: {self.model_path}")
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=torch.float16
            )
            self._model.eval()
            self._model.set_processor(self.model_path)
        return self._model

    def encode(self, image_paths: List[str], batch_size: int = 8,
               show_progress: bool = True) -> List[List[float]]:
        """
        批量图片向量化

        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            向量列表
        """
        embeddings = []
        total_batches = (len(image_paths) - 1) // batch_size + 1

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="图片向量化", total=total_batches)

        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(images=batch_paths)
                embeddings.extend(batch_embeddings.cpu().numpy())

        return embeddings

    def encode_single(self, image_path: str) -> List[float]:
        """
        单个图片向量化

        Args:
            image_path: 图片路径

        Returns:
            向量
        """
        with torch.no_grad():
            embedding = self.model.encode(images=[image_path])
            return embedding.cpu().numpy()[0].tolist()

    @staticmethod
    def load_image_as_base64(image_path: str) -> str:
        """读取图片并转换为 Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def calculate_md5(file_path: str) -> str:
        """计算文件的 MD5 哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @classmethod
    def is_supported_image(cls, file_path: str) -> bool:
        """检查是否为支持的图片格式"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    def get_images_info(self, root_directory: str,
                        include_base64: bool = True) -> List[Dict[str, Any]]:
        """
        获取目录下所有图片的信息

        Args:
            root_directory: 根目录
            include_base64: 是否包含 Base64 编码

        Returns:
            图片信息列表
        """
        result = []

        for subdir in os.listdir(root_directory):
            subdir_path = os.path.join(root_directory, subdir)

            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)

                    if self.is_supported_image(file_path):
                        image_name = subdir  # 使用子目录名作为类别名
                        image_info = {
                            "image_name": image_name,
                            "image_path": file_path,
                            "image_md5": self.calculate_md5(file_path)
                        }

                        if include_base64:
                            image_info["image_base64"] = self.load_image_as_base64(file_path)

                        result.append(image_info)

        return result