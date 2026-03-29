"""多媒体问答路由模块"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services import vllm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["media"])


@router.post("/wenda/image", summary="图片问答接口")
async def image_qa(
    media: UploadFile = File(..., description="图片文件"),
    user_text: str = Form(..., description="用户问题"),
):
    """
    图片问答接口

    Args:
        media: 上传的图片文件
        user_text: 用户关于图片的问题

    Returns:
        模型对图片的回答
    """
    # 验证文件类型
    if not media.content_type or not media.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"文件 '{media.filename}' 不是图片类型，检测到的 Content-Type: {media.content_type}"
        )

    try:
        image_bytes = await media.read()
        answer = await vllm_service.image_qa(image_bytes, user_text)
        return {"description": answer}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"图片问答处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")


@router.post("/wenda/video", summary="视频问答接口")
async def video_qa(
    media: UploadFile = File(..., description="视频文件"),
    user_text: str = Form(..., description="用户问题"),
):
    """
    视频问答接口

    Args:
        media: 上传的视频文件
        user_text: 用户关于视频的问题

    Returns:
        模型对视频的回答
    """
    try:
        video_bytes = await media.read()
        mime_type = media.content_type or "video/mp4"
        answer = await vllm_service.video_qa(video_bytes, mime_type, user_text)
        return {"description": answer}

    except Exception as e:
        logger.error(f"视频问答处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")