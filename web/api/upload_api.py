"""
文件上传 API
"""
import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {
    '.xlsx', '.xls', '.json', '.csv', '.pdf', '.txt', '.doc', '.docx'
}

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到 data/ 目录
    """
    # 检查文件扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 保存文件
    file_path = os.path.join(DATA_DIR, file.filename)

    # 如果文件已存在，添加数字后缀
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file.filename)
        counter = 1
        while os.path.exists(os.path.join(DATA_DIR, f"{base}_{counter}{ext}")):
            counter += 1
        file_path = os.path.join(DATA_DIR, f"{base}_{counter}{ext}")

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"文件上传成功: {file_path}")
        return JSONResponse({
            "success": True,
            "message": f"文件 {file.filename} 上传成功",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        })
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.get("/files")
async def list_files():
    """
    获取 data/ 目录中的文件列表
    """
    try:
        files = []
        for f in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    files.append({
                        "name": f,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "ext": ext,
                        "modified": os.path.getmtime(file_path),
                    })

        # 按修改时间倒序排列
        files.sort(key=lambda x: x["modified"], reverse=True)

        return JSONResponse({"success": True, "files": files})
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """
    删除 data/ 目录中的文件
    """
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件 {file_name} 不存在")

    try:
        os.remove(file_path)
        logger.info(f"文件删除成功: {file_path}")
        return JSONResponse({"success": True, "message": f"文件 {file_name} 已删除"})
    except Exception as e:
        logger.error(f"文件删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
