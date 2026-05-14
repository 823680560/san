import sqlite3
import os
import hashlib
import logging
import multiprocessing as mp
import pandas as pd
from typing import Optional, Dict
from functools import lru_cache

import jieba

# FTS5 索引的 10 个文本列（导入时需做 jieba 分词预处理）
FTS5_TEXT_COLUMNS = [
    "项目名称", "招标单位", "招标代理", "采购主体", "项目类型",
    "项目状态", "中标公司", "城乡分类", "备注", "是否PPP",
    "省份", "城市", "区县",
]

# Excel 列名 → SQLite 列名映射
COLUMN_MAP = {
    "大区": "大区",
    "省份": "省份",
    "城市": "城市",
    "区县": "区县",
    "城乡分类": "城乡分类",
    "项目名称": "项目名称",
    "招标单位": "招标单位",
    "招标代理": "招标代理",
    "采购主体": "采购主体",
    "是否PPP": "是否PPP",
    "项目类型": "项目类型",
    "项目状态": "项目状态",
    "中标时间": "中标时间",
    "年化金额（元）": "年化金额_元",
    "年限（年）": "年限_年",
    "合同金额（元）": "合同金额_元",
    "时间节点": "时间节点",
    "在运营年化金额（元）": "在运营年化金额_元",
    "剩余合同期限（月）": "剩余合同期限_月",
    "预计合同到期时间": "预计合同到期时间",
    "中标公司": "中标公司",
    "清扫面积（㎡）": "清扫面积_㎡",
    "单价（元/㎡）": "单价_元_㎡",
    "转运或处理补贴（元/吨）": "转运或处理补贴_元_吨",
    "转运或处理量（吨/年）": "转运或处理量_吨_年",
    "转运站个数（个）": "转运站个数_个",
    "备注": "备注",
}

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sanitation_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    大区 TEXT, 省份 TEXT, 城市 TEXT, 区县 TEXT,
    城乡分类 TEXT, 项目名称 TEXT, 招标单位 TEXT,
    招标代理 TEXT, 采购主体 TEXT, 是否PPP TEXT,
    项目类型 TEXT, 项目状态 TEXT, 中标时间 DATE,
    年化金额_元 REAL, 年限_年 REAL, 合同金额_元 REAL,
    时间节点 DATE, 在运营年化金额_元 REAL,
    剩余合同期限_月 REAL, 预计合同到期时间 DATE,
    中标公司 TEXT, 清扫面积_㎡ TEXT, 单价_元_㎡ TEXT,
    转运或处理补贴_元_吨 TEXT, 转运或处理量_吨_年 TEXT,
    转运站个数_个 TEXT, 备注 TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS sanitation_projects_fts USING fts5(
    项目名称, 招标单位, 招标代理, 采购主体, 项目类型,
    项目状态, 中标公司, 城乡分类, 备注, 是否PPP,
    省份, 城市, 区县,
    tokenize='unicode61'
);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    """获取 SQLite 连接（允许跨线程使用）"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_file_mtime_hash(excel_path: str) -> str:
    """计算 Excel 文件的 mtime + size 哈希，用于判断是否需重新导入"""
    stat = os.stat(excel_path)
    return hashlib.md5(f"{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()


def _segment_text(text) -> str:
    """用 jieba 对单条文本分词，空格连接"""
    if pd.isna(text):
        return ""
    return " ".join(jieba.cut(str(text)))


def _segment_row(row_values):
    """对单行 FTS 列数据做 jieba 分词（模块级函数，供 multiprocessing 调用）"""
    import jieba
    return [" ".join(jieba.cut(v)) for v in row_values]


def init_project_db(db_path: str) -> sqlite3.Connection:
    """建表 + 建 FTS5 索引"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _get_connection(db_path)
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()
    return conn


def import_excel_to_sqlite(excel_path: str, db_path: str) -> int:
    """读取 xlsx 写入 SQLite（检查 mtime 避免重复导入）

    FTS5 改为 standalone 模式，写入时用 jieba 对中文文本列做分词预处理，
    使得 MATCH 查询能实现词语级搜索而非单字搜索。

    Returns:
        导入的行数（0 表示已是最新，无需导入）
    """
    # 检查是否需要重新导入
    current_hash = _get_file_mtime_hash(excel_path)
    hash_file = db_path + ".hash"
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash and os.path.exists(db_path):
            try:
                conn_check = _get_connection(db_path)
                count = conn_check.execute("SELECT COUNT(*) FROM sanitation_projects").fetchone()[0]
                conn_check.close()
                if count > 0:
                    return 0
            except Exception:
                pass

    # 读取 Excel
    df = pd.read_excel(excel_path, dtype={"中标时间": str, "时间节点": str, "预计合同到期时间": str})

    # 列名映射
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df_sql = df.rename(columns=rename_map)

    # 只保留映射后的列
    target_cols = [v for v in COLUMN_MAP.values() if v in df_sql.columns]
    df_sql = df_sql[target_cols]

    # 清洗城市、区县列的前导/后置空格（Excel 源数据中城市名前常有空格）
    for col in ["省份", "城市", "区县"]:
        if col in df_sql.columns:
            df_sql[col] = df_sql[col].astype(str).str.strip()

    # 写入 SQLite
    conn = _get_connection(db_path)

    # 重建表（兼容旧 external content 格式）
    conn.execute("DROP TABLE IF EXISTS sanitation_projects_fts")
    conn.execute("DROP TABLE IF EXISTS sanitation_projects")
    conn.executescript(SQLITE_SCHEMA)

    # 1. 写入原始数据到内容表
    df_sql.to_sql("sanitation_projects", conn, if_exists="append", index=False)

    # 2. 获取自增 ID
    ids = conn.execute("SELECT id FROM sanitation_projects ORDER BY id").fetchall()
    id_list = [row[0] for row in ids]

    # 3. jieba 分词预处理并写入 FTS5
    col_list_str = ", ".join(FTS5_TEXT_COLUMNS)
    placeholders = ", ".join(["?"] * (1 + len(FTS5_TEXT_COLUMNS)))
    insert_sql = f"INSERT INTO sanitation_projects_fts(rowid, {col_list_str}) VALUES ({placeholders})"

    # 预加载 jieba 词典（首次调用会加载，后续加速）
    _ = len(jieba.lcut("预加载词典"))

    # 提取 FTS 文本列，填充空值后转为 tuple 列表（供多进程使用）
    fts_cols = [c for c in FTS5_TEXT_COLUMNS if c in df_sql.columns]
    fts_data = df_sql[fts_cols].fillna("").astype(str)
    cols_data = [tuple(row) for _, row in fts_data.iterrows()]

    # 多进程并行 jieba 分词
    n_procs = max(1, min(mp.cpu_count() // 2, 8))
    logging.getLogger(__name__).info(
        "FTS5 jieba 分词开始（%d 行，%d 进程）", len(cols_data), n_procs
    )
    with mp.Pool(n_procs) as pool:
        all_seg_vals = pool.map(_segment_row, cols_data, chunksize=2000)

    # 写入 FTS5
    chunk_size = 500
    for chunk_start in range(0, len(df_sql), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df_sql))
        rows = []
        for i in range(chunk_start, chunk_end):
            rows.append([id_list[i]] + all_seg_vals[i])
        conn.executemany(insert_sql, rows)
        conn.commit()

    # 保存哈希
    with open(hash_file, "w") as f:
        f.write(current_hash)

    count = conn.execute("SELECT COUNT(*) FROM sanitation_projects").fetchone()[0]
    conn.close()
    return count


def execute_sqlite(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """执行任意 SQLite SQL，返回 DataFrame"""
    return pd.read_sql(sql, conn)


@lru_cache(maxsize=1)
def get_connection(db_path: str) -> sqlite3.Connection:
    """获取/缓存连接"""
    conn = _get_connection(db_path)
    return conn
