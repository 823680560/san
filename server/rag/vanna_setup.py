import duckdb
import pandas as pd
from typing import Tuple
import os
import time
from contextlib import contextmanager

from server.rag.vanna_llm import DeepseekVanna

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS sanitation_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "大区" TEXT, "省份" TEXT, "城市" TEXT, "区县" TEXT,
    "城乡分类" TEXT, "项目名称" TEXT, "招标单位" TEXT,
    "招标代理" TEXT, "采购主体" TEXT, "是否PPP" TEXT,
    "项目类型" TEXT, "项目状态" TEXT, "中标时间" DATE,
    "年化金额_元" REAL, "年限_年" REAL, "合同金额_元" REAL,
    "时间节点" DATE, "在运营年化金额_元" REAL,
    "剩余合同期限_月" REAL, "预计合同到期时间" DATE,
    "中标公司" TEXT, "清扫面积_㎡" REAL, "单价_元_㎡" REAL,
    "转运或处理补贴_元_吨" REAL, "转运或处理量_吨_年" REAL,
    "转运站个数_个" INTEGER, "备注" TEXT
);
"""

DUCKDB_DDL = """
CREATE TABLE sanitation_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "大区" TEXT, "省份" TEXT, "城市" TEXT, "区县" TEXT,
    "城乡分类" TEXT, "项目名称" TEXT, "招标单位" TEXT,
    "招标代理" TEXT, "采购主体" TEXT, "是否PPP" TEXT,
    "项目类型" TEXT, "项目状态" TEXT, "中标时间" DATE,
    "年化金额_元" REAL, "年限_年" REAL, "合同金额_元" REAL,
    "时间节点" DATE, "在运营年化金额_元" REAL,
    "剩余合同期限_月" REAL, "预计合同到期时间" DATE,
    "中标公司" TEXT, "清扫面积_㎡" REAL, "单价_元_㎡" REAL,
    "转运或处理补贴_元_吨" REAL, "转运或处理量_吨_年" REAL,
    "转运站个数_个" INTEGER, "备注" TEXT
);
"""

SQLITE_TRAINING_PAIRS = [
    # ======== 单条件精确查询 ========
    ("浙江省的环卫项目有哪些",
     'SELECT * FROM sanitation_projects WHERE "省份" = \'浙江\' LIMIT 20'),
    ("PPP模式的项目",
     'SELECT * FROM sanitation_projects WHERE "是否PPP" != \'否\' ORDER BY "项目名称" LIMIT 20'),
    ("政府采购的项目",
     'SELECT * FROM sanitation_projects WHERE "采购主体" = \'政府采购\' LIMIT 20'),
    ("华东地区的项目",
     'SELECT * FROM sanitation_projects WHERE "大区" = \'华东\' LIMIT 20'),
    ("水域保洁的项目",
     'SELECT * FROM sanitation_projects WHERE "项目类型" = \'水域保洁\' LIMIT 20'),
    ("正在进行的项目",
     'SELECT * FROM sanitation_projects WHERE "项目状态" = \'进行中\' LIMIT 20'),
    ("招标公告阶段的项目",
     'SELECT * FROM sanitation_projects WHERE "项目状态" = \'招标公告\' LIMIT 20'),
    ("属于BOT模式的项目",
     'SELECT * FROM sanitation_projects WHERE "项目类型" = \'BOT\' LIMIT 20'),

    # ======== 多条件组合查询 ========
    ("广东省深圳市环卫项目",
     'SELECT * FROM sanitation_projects WHERE "省份" = \'广东\' AND "城市" = \'深圳\' LIMIT 20'),
    ("广东深圳正在进行的环卫项目",
     'SELECT * FROM sanitation_projects WHERE "省份" = \'广东\' AND "城市" = \'深圳\' AND "项目状态" = \'进行中\' LIMIT 20'),
    ("深圳市每个区的环卫项目",
     'SELECT * FROM sanitation_projects WHERE "省份" = \'广东\' AND "城市" = \'深圳\' ORDER BY "区县" LIMIT 30'),
    ("上海的中标公告项目",
     'SELECT * FROM sanitation_projects WHERE "省份" = \'上海\' AND "项目状态" = \'中标公告\' LIMIT 20'),
    ("北京的垃圾分类项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'垃圾\' AND "省份" = \'北京\' LIMIT 20'),

    # ======== 金额范围查询 ========
    ("预算超过500万的项目",
     'SELECT * FROM sanitation_projects WHERE "年化金额_元" > 5000000 ORDER BY "年化金额_元" DESC LIMIT 20'),
    ("年化金额在100万到500万之间的项目",
     'SELECT * FROM sanitation_projects WHERE "年化金额_元" BETWEEN 1000000 AND 5000000 ORDER BY "年化金额_元" DESC LIMIT 20'),
    ("合同金额超过1000万的项目",
     'SELECT * FROM sanitation_projects WHERE "合同金额_元" > 10000000 ORDER BY "合同金额_元" DESC LIMIT 20'),
    ("合同金额低于50万的项目",
     'SELECT * FROM sanitation_projects WHERE "合同金额_元" < 500000 ORDER BY "合同金额_元" DESC LIMIT 20'),
    ("年限超过5年的项目",
     'SELECT * FROM sanitation_projects WHERE "年限_年" > 5 ORDER BY "年限_年" DESC LIMIT 20'),

    # ======== 日期范围查询 ========
    ("2023年的中标项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" BETWEEN \'2023-01-01\' AND \'2023-12-31\' ORDER BY "中标时间" DESC LIMIT 20'),
    ("2022年以后中标的项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" >= \'2022-01-01\' ORDER BY "中标时间" DESC LIMIT 20'),
    ("2020年到2022年之间的项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" BETWEEN \'2020-01-01\' AND \'2022-12-31\' ORDER BY "中标时间" DESC LIMIT 20'),
    ("2024年到期的项目",
     'SELECT * FROM sanitation_projects WHERE "预计合同到期时间" BETWEEN \'2024-01-01\' AND \'2024-12-31\' ORDER BY "预计合同到期时间" DESC LIMIT 20'),

    # ======== 排除/空值查询 ========
    ("不是PPP模式的项目",
     'SELECT * FROM sanitation_projects WHERE "是否PPP" IS NULL OR "是否PPP" = \'\' OR "是否PPP" = \'否\' LIMIT 20'),
    ("没有填写中标时间的项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" IS NULL OR "中标时间" = \'\' LIMIT 20'),
    ("城乡分类为空白的项目",
     'SELECT * FROM sanitation_projects WHERE "城乡分类" IS NULL OR "城乡分类" = \'\' LIMIT 20'),

    # ======== 模糊搜索（FTS5） ========
    ("找水域保洁相关的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'水域 OR 保洁\' LIMIT 20'),
    ("搜索河道清理的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'河道 OR 清理\' LIMIT 20'),
    ("查一下中环洁公司的中标情况",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'中环 洁\' LIMIT 20'),
    ("有哪些垃圾处理项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'垃圾 OR 处理\' LIMIT 20'),
    ("搜索道路清扫项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'道路 OR 清扫\' LIMIT 20'),
    ("搜索垃圾分类相关的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'垃圾 OR 分类\' LIMIT 20'),
    ("搜索物业相关的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'物业\' LIMIT 20'),
    ("找转运处理相关的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'转运 OR 处理\' LIMIT 20'),

    # ======== 公司名 + 地区组合搜索（FTS5 已包含省份/城市/区县列） ========
    ("查一下粤丰粤展在保定的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'粤 丰粤展 保定\' LIMIT 20'),
    ("查一下北控环境在深圳的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'北控 环境 深圳\' LIMIT 20'),
    ("中联重科在湖南省中了哪些标",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'中联重科 湖南\' LIMIT 20'),

    # ======== 按项目名称模糊搜索（FTS5） ========
    ("园岭街道保湿保洁项目的中标公司是谁",
 'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'园岭 街道 保湿 保洁\' LIMIT 10'),
    ("株洲南部生活垃圾焚烧发电PPP项目的服务周期和到期时间",
 'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'株洲 南部 生活 垃圾 焚烧 PPP\' LIMIT 10'),
    ("淳安县千岛湖城区路灯保洁项目的中标信息",
 'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'淳安 千岛湖 路灯 保洁\' LIMIT 10'),
    ("七宝镇绿化垃圾处置项目的中标公司",
 'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'七宝镇 绿化 垃圾 处置\' LIMIT 10'),

    # ======== FTS5 关键词搜索（多样化措辞） ========
    ("包含绿化的项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'绿化\' LIMIT 10'),
    ("项目名称或中标公司包含绿化关键词的",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'绿化\' LIMIT 10'),
    ("项目名称中有分类两个字的",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'分类\' LIMIT 10'),
    ("找一下跟河道清理相关的环卫项目",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'河道 OR 清理\' LIMIT 10'),
    ("项目名称或中标公司中包含关键词的",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'道路 保洁\' LIMIT 10'),
    ("根据项目名称的关键字搜索",
     'SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'道路 OR 清扫\' LIMIT 10'),
    # 仅当用户明确指定列名时，才在 FTS 中使用 sp. 前缀选特定列
    ("项目名称中包含分类，显示项目名称、中标公司、中标时间",
     'SELECT sp."项目名称", sp."中标公司", sp."中标时间" FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH \'分类\' LIMIT 10'),

    # ======== 日期"之后"语义（>=） ========
    ("2018年8月4日之后最早中标的项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" >= \'2018-08-04\' ORDER BY "中标时间" ASC LIMIT 10'),
    ("2020年3月之后中标的项目",
     'SELECT * FROM sanitation_projects WHERE "中标时间" >= \'2020-03-01\' ORDER BY "中标时间" ASC LIMIT 20'),
    
    # ======== 更复杂的日期查询 ========
    ("2023年之后中标的北京项目",
 'SELECT * FROM sanitation_projects WHERE "省份" = \'北京\' AND "中标时间" >= \'2023-01-01\' ORDER BY "中标时间" ASC LIMIT 20'),
    ("中标时间最近3个月的项目",
 'SELECT * FROM sanitation_projects WHERE "中标时间" >= date(\'now\', \'-3 months\') ORDER BY "中标时间" ASC LIMIT 20'),

    # ======== 列名引号规范示范 ========
    ("保定市的环卫服务项目",
     'SELECT * FROM sanitation_projects WHERE "城市" = \'保定市\' LIMIT 10'),
    ("深圳正在进行的项目",
     'SELECT * FROM sanitation_projects WHERE "城市" = \'深圳市\' AND "项目状态" = \'进行中\' LIMIT 20'),
    ("城乡分类为一城乡一体化的项目",
     'SELECT * FROM sanitation_projects WHERE "城乡分类" = \'城乡一体化\' LIMIT 10'),

    # ======== 排序/最新查询 ========
    ("最新的中标项目有哪些",
     'SELECT * FROM sanitation_projects WHERE "中标时间" IS NOT NULL ORDER BY "中标时间" DESC LIMIT 20'),
    ("合同金额最高的项目",
     'SELECT * FROM sanitation_projects WHERE "合同金额_元" IS NOT NULL ORDER BY "合同金额_元" DESC LIMIT 10'),
    ("合同年限超过14年的项目中，年限最长的项目",
     'SELECT * FROM sanitation_projects WHERE "年限_年" > 14 AND "年限_年" IS NOT NULL ORDER BY "年限_年" DESC LIMIT 10'),
    ("年限最长的前10个项目",
     'SELECT * FROM sanitation_projects WHERE "年限_年" IS NOT NULL ORDER BY "年限_年" DESC LIMIT 10'),
    ("合同金额最高的前10个项目是哪些",
     'SELECT * FROM sanitation_projects WHERE "合同金额_元" IS NOT NULL ORDER BY "合同金额_元" DESC LIMIT 10'),
]

DUCKDB_TRAINING_PAIRS = [
    ("哪些省份的环卫项目数量最多，排名前十",
     'SELECT "省份", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "省份" IS NOT NULL AND "省份" != \'\' GROUP BY "省份" ORDER BY "项目数" DESC LIMIT 10'),    
    ("平均年化金额最高的前10种项目类型",
     'SELECT "项目类型", AVG("年化金额_元") AS "平均年化" FROM sanitation_projects WHERE "年化金额_元" IS NOT NULL AND "项目类型" IS NOT NULL GROUP BY "项目类型" ORDER BY "平均年化" DESC LIMIT 10'),    
    ("各省份项目数量排名",
     'SELECT "省份", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "省份" IS NOT NULL AND "省份" != \'\' GROUP BY "省份" ORDER BY "项目数" DESC LIMIT 50'), 
    ("每年项目数量和合同总额",
     'SELECT strftime(\'%Y\', "中标时间") AS "年份", COUNT(*) AS "项目数", SUM("合同金额_元") AS "总金额" FROM sanitation_projects WHERE "中标时间" IS NOT NULL GROUP BY "年份" ORDER BY "年份"'),    
    ("各省份各类型的项目分布",
     'SELECT "省份", "项目类型", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "省份" IS NOT NULL AND "省份" != \'\' AND "项目类型" IS NOT NULL GROUP BY "省份", "项目类型" ORDER BY "省份", "项目数" DESC LIMIT 200'),    
    ("合同金额最高的10个项目",
     'SELECT * FROM sanitation_projects WHERE "合同金额_元" IS NOT NULL ORDER BY "合同金额_元" DESC LIMIT 10'),    
    ("城乡分类的项目数量对比",
     'SELECT "城乡分类", COUNT(*) AS "项目数", AVG("年化金额_元") AS "平均金额" FROM sanitation_projects WHERE "年化金额_元" IS NOT NULL AND "城乡分类" IS NOT NULL AND "城乡分类" != \'\' GROUP BY "城乡分类"'),    
    ("各区域合同总金额排名",
     'SELECT "大区", SUM("合同金额_元") AS "总金额" FROM sanitation_projects WHERE "合同金额_元" IS NOT NULL AND "大区" IS NOT NULL AND "大区" != \'\' GROUP BY "大区" ORDER BY "总金额" DESC'),    
    ("平均合同年限按项目类型",
     'SELECT "项目类型", AVG("年限_年") AS "平均年限", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "年限_年" IS NOT NULL AND "项目类型" IS NOT NULL GROUP BY "项目类型" ORDER BY "平均年限" DESC'),
    ("每年合同金额趋势",
     'SELECT strftime(\'%Y\', "中标时间") AS "年份", SUM("年化金额_元") AS "总年化" FROM sanitation_projects WHERE "中标时间" IS NOT NULL GROUP BY "年份" ORDER BY "年份"'),    
    ("各城市项目数量TOP20",
     'SELECT "省份", "城市", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "城市" IS NOT NULL AND "省份" IS NOT NULL AND "省份" != \'\' GROUP BY "省份", "城市" ORDER BY "项目数" DESC LIMIT 20'),
    ("每年项目数量趋势",
     'SELECT strftime(\'%Y\', "中标时间") AS "年份", COUNT(*) AS "项目数量" FROM sanitation_projects WHERE "中标时间" IS NOT NULL GROUP BY "年份" ORDER BY "年份"'),    
    ("合同金额最大的省份排名",
     'SELECT "省份", SUM("合同金额_元") AS "总合同额" FROM sanitation_projects WHERE "合同金额_元" IS NOT NULL AND "省份" IS NOT NULL AND "省份" != \'\' GROUP BY "省份" ORDER BY "总合同额" DESC LIMIT 10'),    
    ("各项目状态的数量分布",
     'SELECT "项目状态", COUNT(*) AS "项目数" FROM sanitation_projects WHERE "项目状态" IS NOT NULL AND "项目状态" != \'\' GROUP BY "项目状态" ORDER BY "项目数" DESC'),        
    ("各省份的年化金额总和",
     'SELECT "省份", SUM("年化金额_元") AS "年化总金额" FROM sanitation_projects WHERE "年化金额_元" IS NOT NULL AND "省份" IS NOT NULL AND "省份" != \'\' GROUP BY "省份" ORDER BY "年化总金额" DESC LIMIT 50'),    
    ("每年的平均合同年限",
     'SELECT strftime(\'%Y\', "中标时间") AS "年份", AVG("年限_年") AS "平均年限" FROM sanitation_projects WHERE "年限_年" IS NOT NULL GROUP BY "年份" ORDER BY "年份"'),
]


def _train_sqlite_vanna(vanna, sqlite_conn):
    """训练 SQLite Vanna 实例"""
    # 获取实际数据条数作为文档
    count = sqlite_conn.execute("SELECT COUNT(*) FROM sanitation_projects").fetchone()[0]

    vanna.train(ddl=SQLITE_DDL)
    vanna.train(documentation=f"sanitation_projects 表共有 {count} 条环卫项目记录")
    vanna.train(documentation="年化金额_元表示每年合同金额，单位元，取值范围 600~520,000,000")
    vanna.train(documentation="合同金额_元表示合同总金额，单位元，取值范围 1~7,965,000,000")
    vanna.train(documentation="年限_年表示合同年限，可为小数如 0.5 表示半年")
    vanna.train(documentation="中标时间格式为 YYYY-MM-DD HH:MM:SS，可用 >=、<=、BETWEEN 筛选，数据范围 2015~2024 年")
    vanna.train(documentation="预计合同到期时间格式为 YYYY-MM-DD HH:MM:SS")
    vanna.train(documentation="项目状态常见值: '招标公告'、'中标公告'、'进行中'、'资格预审'、'废标公告'、'政府采购'、'招标预告'、'更正公告'、'项目前期'、'预审结果'")
    vanna.train(documentation="项目类型常见值: '垃圾分类'、'垃圾收运'、'垃圾处理'、'道路清扫'、'水域保洁'、'公厕保洁'、'公园管养'、'BOT'、'绿化'、'物业'、'转运'")
    vanna.train(documentation="省份字段存完整的省份名称如 '上海'、'广东'、'浙江'，不加'省'字。城市字段存城市名称如 '保定市'、'深圳市'，不带前导空格")
    vanna.train(documentation="大区字段值: '华东'、'华南'、'华北'、'华中'、'西南'、'西北'、'东北'")
    vanna.train(documentation="FTS5 模糊搜索语法：WHERE sanitation_projects_fts MATCH '关键词'（注意：WHERE 中必须使用 FTS 表全名 sanitation_projects_fts，严禁使用别名 fts，不能出现WHERE fts MATCH的情况；别名fts只能在 JOIN 中使用）")
    vanna.train(documentation="FTS5 中文文本已用 jieba 分词预处理，多关键词可用空格 AND 连接：MATCH '道路 保洁' 匹配同时包含'道路'和'保洁'的项目")
    vanna.train(documentation=(
        "【CRITICAL - 强制执行】关键词搜索规则：\n"
        "当用户问题中包含以下任何关键词时，必须使用 FTS5 MATCH，禁止使用 LIKE：\n"
        "- '包含'（如'包含绿化'、'包含分类'）\n"
        "- '关键词'（如'绿化关键词'）\n"
        "- '搜索'（如'搜索河道清理'）\n"
        "- '找'（如'找一下转运处理相关的'）\n"
        "- '查'（如'查一下中环洁公司'）\n"
        "- 任何业务词：绿化、分类、清扫、垃圾、保洁、转运、处理、河道、物业、道路\n\n"
        "正确示例：\n"
        "Q: '项目名称中包含\"分类\"这两个字的项目'\n"
        "SQL: SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp "
        "ON fts.rowid = sp.id WHERE sanitation_projects_fts MATCH '分类' LIMIT 10\n\n"
        "错误示例（禁止）：\n"
        "SELECT * FROM sanitation_projects WHERE \"项目名称\" LIKE '%分类%'\n"
        "即使要求返回部分列，也必须使用 FTS，只是 SELECT 中加 sp. 前缀。"
    ))
    vanna.train(documentation=(
        "【CRITICAL】FTS5 选择特定列规则：如果用户要求返回特定列（如'显示项目名称、中标公司、中标时间'），"
        "仍然必须使用 FTS5 MATCH，但 SELECT 中需要用 sp. 前缀限定列名以避免歧义。\n"
        "正确写法（FTS + 指定列）：\n"
        "SELECT sp.\"项目名称\", sp.\"中标公司\", sp.\"中标时间\"\n"
        "FROM sanitation_projects_fts fts\n"
        "JOIN sanitation_projects sp ON fts.rowid = sp.id\n"
        "WHERE sanitation_projects_fts MATCH '分类' LIMIT 10\n"
        "错误写法（会报错）：SELECT \"项目名称\" FROM ...（未加 sp. 前缀导致 ambiguous column name）\n"
        "不要为了指定列就改用非 FTS 查询。"
    ))
    vanna.train(documentation=(
        "【CRITICAL】列名引用规则：所有中文列名必须用双引号括起来，例如：\n"
        "SELECT * FROM sanitation_projects WHERE \"中标时间\" >= '2018-08-04' ORDER BY \"中标时间\"\n"
        "SELECT * FROM sanitation_projects WHERE \"城市\" = '保定市'\n"
        "字符串常量用单引号，列名用双引号，切勿混淆。"
    ))
    vanna.train(documentation=(
        "【CRITICAL】日期语义规则：\n"
        "- 中文'之后'/'以后'/'以来'在日期筛选中包含指定日期，应使用 >= 而非 >。\n"
        "  例如：'2018年8月4日之后' → \"中标时间\" >= '2018-08-04'\n"
        "  例如：'2022年以后' → \"中标时间\" >= '2022-01-01'\n"
        "- 中文'之前'/'以前'应使用 <= 而非 <。\n"
        "- 中文'超过'/'大于'应使用 > 而非 >=。"
    ))
    vanna.train(documentation=(
    "【具体项目名称查询】\n"
    "当用户提到具体的项目名称时（如'园岭街道保湿保洁项目'），也使用 FTS5 MATCH 进行查询：\n"
    "SELECT sp.* FROM sanitation_projects_fts fts JOIN sanitation_projects sp ON fts.rowid = sp.id "
    "WHERE sanitation_projects_fts MATCH '园岭 街道 保湿 保洁' LIMIT 10\n"
    "将项目名称中的关键词用空格分隔进行匹配。"
    ))
    vanna.train(documentation="是否PPP字段：值为'是'或'否'或空，查询非PPP项目时需同时判断 IS NULL、空字符串和'否'三种情况")
    vanna.train(documentation="城乡分类常见值: '城乡一体化'、'城区'、'乡镇'、'农村'、'其他'")
    vanna.train(documentation=(
    "【CRITICAL】排序规则：\n"
    "- 问题中包含'最长'、'最高'、'最大'、'最多' → 必须加 ORDER BY [字段] DESC\n"
    "- 问题中包含'最短'、'最低'、'最小'、'最少' → 必须加 ORDER BY [字段] ASC\n"
    "- 问题中包含'最早' → 必须加 ORDER BY [时间字段] ASC\n"
    "- 问题中包含'最新'、'最近' → 必须加 ORDER BY [时间字段] DESC\n"
    "- 使用 LIMIT 时通常需要配合 ORDER BY，否则结果不确定"
    ))
    vanna.train(documentation=(
    "【ABO/PPP 模式查询】\n"
    "ABO 模式和 PPP 模式都存储在 '是否PPP' 字段中。\n"
    "✅ 正确：WHERE \"是否PPP\" = 'ABO'\n"
    "✅ 正确：WHERE \"是否PPP\" = 'PPP'\n"
    "❌ 错误：WHERE \"项目类型\" = 'ABO'\n"
    "❌ 错误：WHERE \"项目类型\" = 'PPP'"
    ))
    

    for question, sql in SQLITE_TRAINING_PAIRS:
        vanna.train(question=question, sql=sql)


def _train_duckdb_vanna(vanna, duckdb_conn):
    """训练 DuckDB Vanna 实例"""
    count = duckdb_conn.execute("SELECT COUNT(*) FROM sanitation_projects").fetchone()[0]    
    # 动态获取金额范围
    min_amount = duckdb_conn.execute(
        "SELECT MIN(年化金额_元) FROM sanitation_projects WHERE 年化金额_元 IS NOT NULL"
    ).fetchone()[0]
    max_amount = duckdb_conn.execute(
        "SELECT MAX(年化金额_元) FROM sanitation_projects WHERE 年化金额_元 IS NOT NULL"
    ).fetchone()[0]    
    # 动态获取日期范围
    date_range = duckdb_conn.execute(
        "SELECT MIN(中标时间), MAX(中标时间) FROM sanitation_projects WHERE 中标时间 IS NOT NULL"
    ).fetchone()
    min_date = str(date_range[0])[:4] if date_range[0] else "未知"
    max_date = str(date_range[1])[:4] if date_range[1] else "未知"
    vanna.train(ddl=DUCKDB_DDL)
    vanna.train(documentation=f"sanitation_projects 表共有 {count} 条环卫项目记录")
    vanna.train(documentation=f"年化金额_元表示每年合同金额，单位元，范围 {min_amount:,.0f}~{max_amount:,.0f} 元")
    vanna.train(documentation=f"合同金额_元表示合同总金额，单位元")
    vanna.train(documentation=f"中标时间数据范围 {min_date}~{max_date} 年")    
    vanna.train(documentation=(
        "【DuckDB 日期函数语法 - 重要】\n"
        "DuckDB 的 strftime 语法：strftime(格式, 时间字段)\n"
        "✅ 正确：strftime('%Y', \"中标时间\")\n"
        "❌ 错误：strftime(\"中标时间\", '%Y')\n"
        "常见格式：'%Y' 年份, '%m' 月份, '%d' 日, '%Y-%m' 年月"
    ))    
    vanna.train(documentation=(
        "项目状态常见值: '招标公告'、'中标公告'、'进行中'、'资格预审'"
    ))
    vanna.train(documentation=(
        "省份字段存完整的省份名称如 '上海'、'广东'，不加'省'字"
    ))
    vanna.train(documentation=(
        "大区字段值: '华东'、'华南'、'华北'、'华中'、'西南'、'西北'、'东北'"
    ))
    vanna.train(documentation=(
        "【CRITICAL】列名引用规则：所有中文列名必须用双引号括起来，"
        "例如：SELECT * FROM sanitation_projects WHERE \"省份\" IS NOT NULL GROUP BY \"省份\""
    ))
    vanna.train(documentation=(
        "【CRITICAL】聚合查询中，空值无统计意义，必须加 IS NOT NULL 过滤，"
        "例如：WHERE \"年化金额_元\" IS NOT NULL；省份统计加 WHERE \"省份\" IS NOT NULL AND \"省份\" != ''"
    ))
    vanna.train(documentation=(
        "【CRITICAL】日期语义规则：中文'之后'/'以后'包含指定日期，使用 >= 而非 >。"
        "'之前'/'以前'使用 <= 而非 <。"
    ))    
    vanna.train(documentation=(
        "【DuckDB 高级功能】\n"
        "- 字符串聚合：string_agg(\"项目名称\", ', ')\n"
        "- 近似计数：approx_count_distinct(\"省份\")\n"
        "- 分位数：quantile_cont(\"合同金额_元\", 0.5) AS \"中位数\""
    ))
    vanna.train(documentation=(
        "【聚合查询规范】\n"
        "在进行 GROUP BY 聚合查询时，如果用户明确指定了筛选条件（如'华东地区'），"
        "必须在 WHERE 子句中添加对应的过滤条件。\n"
        "示例：'各个大区的合同总金额排名，华东地区的总金额是多少'\n"
        "SQL: SELECT \"大区\", SUM(\"合同金额_元\") FROM sanitation_projects "
        "WHERE \"大区\" = '华东' GROUP BY \"大区\""
    ))
    
    for question, sql in DUCKDB_TRAINING_PAIRS:
        vanna.train(question=question, sql=sql)

# 全局缓存
_VANNA_CACHE = None

def safe_run_sql(sql, conn, db_type="sqlite"):
    """安全执行 SQL，返回 DataFrame 或空 DataFrame"""
    try:
        if db_type == "sqlite":
            return pd.read_sql(sql, conn)
        else:
            return conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"[SQL错误 - {db_type}] {sql[:200]}...\n错误信息: {e}")
        return pd.DataFrame()

def _train_with_timing(vanna, train_func, name, *args):
    """带计时的训练包装器"""
    print(f"[训练] {name} 开始训练...")
    start = time.time()
    train_func(vanna, *args)
    elapsed = time.time() - start
    print(f"[训练] {name} 完成，耗时 {elapsed:.2f} 秒")

def init_vanna_instances(llm_config: dict, sqlite_conn, db_path: str, force_reload=False):
    """创建两个 Vanna 实例并加载训练数据

    Args:
        llm_config: LLM 配置字典 (model, api_key, base_url)
        sqlite_conn: SQLite 连接
        db_path: SQLite 数据库路径（供 DuckDB ATTACH 使用）
        force_reload: 是否强制重新加载（忽略缓存）

    Returns:
        (vanna_sqlite, vanna_duckdb, duckdb_conn)
    """
    global _VANNA_CACHE
    
    # 检查缓存
    if not force_reload and _VANNA_CACHE is not None:
        print("[初始化] 使用缓存的 Vanna 实例")
        return _VANNA_CACHE
    
    from configs.config import EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, resolve_model_path

    # 解析模型路径：本地存在则用本地，否则用 HF 模型 ID 自动下载
    model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)

    # 注入 embedding 配置
    vanna_config = {**llm_config, "embedding_model_path": model_path}
    
    # ---------- SQLite Vanna ----------
    print("[初始化] 创建 SQLite Vanna 实例...")
    vanna_sqlite = DeepseekVanna(config=vanna_config)
    vanna_sqlite.run_sql = lambda sql: safe_run_sql(sql, sqlite_conn, "sqlite")
    _train_with_timing(vanna_sqlite, _train_sqlite_vanna, "SQLite Vanna", sqlite_conn)
    
    # ---------- DuckDB Vanna ----------
    print("[初始化] 创建 DuckDB Vanna 实例...")
    duckdb_conn = duckdb.connect()
    
    # 安全处理路径中的单引号
    safe_path = db_path.replace("'", "''")
    duckdb_conn.execute(f"ATTACH '{safe_path}' AS sqlite_db (TYPE SQLITE);")
    duckdb_conn.execute("USE sqlite_db;")
    
    vanna_duckdb = DeepseekVanna(config=vanna_config)
    vanna_duckdb.run_sql = lambda sql: safe_run_sql(sql, duckdb_conn, "duckdb")
    _train_with_timing(vanna_duckdb, _train_duckdb_vanna, "DuckDB Vanna", duckdb_conn)
    
    # 缓存结果
    _VANNA_CACHE = (vanna_sqlite, vanna_duckdb, duckdb_conn)
    
    print("[初始化] Vanna 实例初始化完成")
    return _VANNA_CACHE

def close_vanna_instances():
    """关闭并清理 Vanna 实例"""
    global _VANNA_CACHE
    if _VANNA_CACHE is not None:
        _, _, duckdb_conn = _VANNA_CACHE
        duckdb_conn.close()
        _VANNA_CACHE = None
        print("[清理] Vanna 实例已关闭")
