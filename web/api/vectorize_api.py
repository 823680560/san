"""
向量化构建 API
"""
import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")

router = APIRouter(tags=["vectorize"])


class VectorizeRequest(BaseModel):
    """向量化请求"""
    file_name: str
    kb_name: Optional[str] = None  # 知识库名称，默认使用文件名
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50


@router.post("/vectorize")
async def vectorize_file(req: VectorizeRequest):
    """
    对指定文件进行向量化构建

    1. 从 data/ 目录读取文件
    2. 解析文件内容
    3. 分片
    4. 构建 FAISS 索引
    5. 保存到 knowledge_base/ 目录
    """
    file_path = os.path.join(DATA_DIR, req.file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件 {req.file_name} 不存在于 data/ 目录")

    # 知识库名称：默认使用文件名（不含扩展名）
    kb_name = req.kb_name or os.path.splitext(req.file_name)[0]
    save_path = os.path.join(KB_DIR, kb_name)

    try:
        # 延迟导入，避免启动时加载所有依赖
        from server.rag.vector_store import build_index_from_file

        logger.info(f"开始向量化: file={req.file_name}, kb={kb_name}, chunk_size={req.chunk_size}")

        index = build_index_from_file(
            file_path=file_path,
            save_path=save_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

        # 获取索引信息
        doc_count = len(index.docstore.docs) if hasattr(index, 'docstore') else 0

        return JSONResponse({
            "success": True,
            "message": f"文件 {req.file_name} 向量化完成",
            "kb_name": kb_name,
            "save_path": save_path,
            "node_count": doc_count,
        })

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"缺少依赖库: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"向量化失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")


@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """
    获取已有知识库列表
    """
    try:
        kb_list = []
        if os.path.exists(KB_DIR):
            for name in os.listdir(KB_DIR):
                kb_path = os.path.join(KB_DIR, name)
                if os.path.isdir(kb_path):
                    # 检查是否是有效的 FAISS 索引目录（查找 faiss 索引文件或 vector_store.json）
                    has_index = any(
                        f.endswith(".faiss") or f.endswith("__vector_store.json")
                        for f in os.listdir(kb_path)
                    )
                    kb_list.append({
                        "name": name,
                        "path": kb_path,
                        "has_index": has_index,
                    })

        return JSONResponse({"success": True, "knowledge_bases": kb_list})
    except Exception as e:
        logger.error(f"获取知识库列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-bases/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    """
    删除知识库
    """
    import shutil
    kb_path = os.path.join(KB_DIR, kb_name)
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail=f"知识库 {kb_name} 不存在")

    try:
        shutil.rmtree(kb_path)
        logger.info(f"知识库删除成功: {kb_path}")
        return JSONResponse({"success": True, "message": f"知识库 {kb_name} 已删除"})
    except Exception as e:
        logger.error(f"知识库删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
