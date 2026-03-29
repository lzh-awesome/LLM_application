"""
视频向量化模块
使用 VideoCLIP-XL 模型进行视频向量化
"""
import os
import base64
import hashlib
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import torch
import numpy as np
import cv2


class VideoEmbedding:
    """视频向量化类"""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

    # 归一化参数
    V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def __init__(self, model_path: str, weights_path: str, device: str = "cuda"):
        """
        初始化视频向量化模型

        Args:
            model_path: VideoCLIP 模型路径
            weights_path: 模型权重路径
            device: 设备
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            print(f"正在加载视频向量化模型: {self.model_path}")
            # 动态导入，避免未安装时报错
            from modeling import VideoCLIP_XL

            self._model = VideoCLIP_XL()
            state_dict = torch.load(self.weights_path, map_location="cpu")
            self._model.load_state_dict(state_dict)
            self._model.cuda().eval()
        return self._model

    def encode(self, video_paths: List[str], batch_size: int = 8,
               frames_per_video: int = 8,
               show_progress: bool = True) -> List[List[float]]:
        """
        批量视频向量化

        Args:
            video_paths: 视频路径列表
            batch_size: 批处理大小
            frames_per_video: 每个视频提取的帧数
            show_progress: 是否显示进度条

        Returns:
            向量列表
        """
        embeddings = []
        total_batches = (len(video_paths) - 1) // batch_size + 1

        iterator = range(0, len(video_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="视频向量化", total=total_batches)

        for i in iterator:
            batch_paths = video_paths[i:i + batch_size]
            batch_tensors = []

            for video_path in batch_paths:
                try:
                    tensor = self._preprocess_video(video_path, frames_per_video)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"⚠️ 处理视频失败 {video_path}: {e}")
                    continue

            if not batch_tensors:
                continue

            batch_tensor = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                features = self.model.vision_model.get_vid_features(batch_tensor).float()
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.extend(features.cpu().numpy())

        return embeddings

    def encode_single(self, video_path: str, frames_per_video: int = 8) -> List[float]:
        """
        单个视频向量化

        Args:
            video_path: 视频路径
            frames_per_video: 提取的帧数

        Returns:
            向量
        """
        tensor = self._preprocess_video(video_path, frames_per_video)
        with torch.no_grad():
            features = self.model.vision_model.get_vid_features(tensor).float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def _preprocess_video(self, video_path: str, num_frames: int = 8) -> torch.Tensor:
        """
        视频预处理

        Args:
            video_path: 视频路径
            num_frames: 提取的帧数

        Returns:
            预处理后的张量
        """
        video = cv2.VideoCapture(video_path)
        frames = []

        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break

        video.release()

        if len(frames) == 0:
            raise ValueError(f"无法从视频中提取帧: {video_path}")

        # 均匀采样帧
        step = max(1, len(frames) // num_frames)
        frames = frames[::step][:num_frames]

        vid_tube = []
        for frame in frames:
            frame = frame[:, :, ::-1]  # BGR to RGB
            frame = cv2.resize(frame, (224, 224))
            frame = np.expand_dims(self._normalize(frame), axis=(0, 1))
            vid_tube.append(frame)

        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))

        return torch.from_numpy(vid_tube).float().cuda()

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化"""
        return (data / 255.0 - self.V_MEAN) / self.V_STD

    @staticmethod
    def load_video_as_base64(video_path: str) -> str:
        """读取视频并转换为 Base64"""
        with open(video_path, "rb") as f:
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
    def is_supported_video(cls, file_path: str) -> bool:
        """检查是否为支持的视频格式"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    def get_videos_info(self, root_directory: str,
                        include_base64: bool = True) -> List[Dict[str, Any]]:
        """
        获取目录下所有视频的信息

        Args:
            root_directory: 根目录
            include_base64: 是否包含 Base64 编码

        Returns:
            视频信息列表
        """
        result = []

        for subdir in os.listdir(root_directory):
            subdir_path = os.path.join(root_directory, subdir)

            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)

                    if self.is_supported_video(file_path):
                        video_name = subdir  # 使用子目录名作为类别名
                        video_info = {
                            "video_name": video_name,
                            "video_path": file_path,
                            "video_md5": self.calculate_md5(file_path)
                        }

                        if include_base64:
                            video_info["video_base64"] = self.load_video_as_base64(file_path)

                        result.append(video_info)

        return result