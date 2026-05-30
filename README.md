# 知识库问答系统

> 基于 RAG（检索增强生成）技术的企业文档智能问答系统

---

##  快速启动

### 最简单方式：双击 start.bat

直接双击 `start.bat` 文件即可启动服务。

### 手动启动

```bash
cd d:\program\0429\AIProjects\project2_knowledge_qa
streamlit run app.py --server.port 5002
```

浏览器自动打开：`http://localhost:5002`

---

##  项目状态

**状态**：已完成，可正常运行

**最新更新**：2026-05-07
-  修复 Streamlit stdout 关闭导致 tqdm 崩溃问题
-  实现 DeepSeek 风格深色主题 UI（蓝紫渐变）
-  优化文件预览交互（默认不显示，点击后显示）
-  修复侧边栏、输入框、按钮等样式问题

---

##  使用方法

### 1. 首次使用
1. 左侧输入 DeepSeek API Key（或确认 `.env` 已配置）
2. 点击「 重建索引」按钮
3. 系统自动加载知识库文档并生成语义向量

### 2. 日常使用
- 左侧「知识库文件」点击文件名可预览内容
- 底部输入框输入问题
- 系统检索知识库相关内容 → 调用 DeepSeek 生成回答

### 3. 添加新文档
1. 将 `.txt` 文件放入 `knowledge_base/` 文件夹
2. 点击「 重建索引」
3. 新文档即可被检索

---

##  UI 设计

**主题风格**：浅灰绿配色主题
- 主背景：浅灰色 `#EFEFEF`
- 侧边栏：深青色渐变 `#315762 → #2A4D54`
- 辅助色：沙色 `#D0BA98`
- 文字色：深棕色 `#23140C`
- 替补色：`#B7D0D4`, `#D6E3E9`

**配色表**：
| 元素 | 颜色代码 | 说明 |
|------|----------|------|
| 主背景 | #EFEFEF | 浅灰色 |
| 侧边栏 | #315762 | 深青色 |
| 辅助色 | #D0BA98 | 沙色（按钮/强调） |
| 文字色 | #23140C | 深棕色 |
| 替补色 | #B7D0D4, #D6E3E9 | 浅青色调 |

**交互优化**：
- 文件预览默认不显示，点击左侧文件后才显示内容
- 聊天输入框独立，不受文件预览影响

---

##  技术架构

| 组件 | 技术 | 来源 |
|------|------|------|
| **前端框架** | Streamlit | 本地运行 |
| **语义检索** | BAAI/bge-small-zh-v1.5 | HuggingFace Mirror |
| **AI 对话** | DeepSeek Chat API | api.deepseek.com |
| **知识库存储** | 本地 TXT 文件 | 不上传云端 |
| **向量处理** | NumPy | 本地计算 |

---

## 已解决的问题

### 问题 1：Streamlit 启动崩溃
**现象**：`ValueError: I/O operation on closed file`

**原因**：Streamlit 关闭 stdout，但 tqdm（transformers 模型加载时使用）尝试写入已关闭的 stdout

**解决**：在 `SemanticModel.load()` 中 patch `sys.stdout.flush`，捕获 ValueError/OSError 异常

### 问题 2：TF-IDF 检索效果差
**现象**：中文检索相似度为 0，什么都找不到

**原因**：
- TF-IDF 按字符级别处理中文，无法理解语义
- 分块策略不当，每个块太短

**解决**：
- 改用语义 Embedding 模型（BAAI/bge-small-zh-v1.5）
- 改进分块策略，保留完整语义段落

### 问题 3：HuggingFace 模型下载失败
**现象**：`WinError 10060` 网络超时

**原因**：国内无法直接访问 huggingface.co

**解决**：使用 hf-mirror.com 镜像（`os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`）

### 问题 4：深色主题样式问题
**现象**：
- 侧边栏有黑色横条影响渐变背景
- API Key 输入框白色背景看不清文字
- 底部聊天输入区白色区域突兀

**解决**：
- CSS 强制覆盖所有容器层背景为蓝黑色
- JS MutationObserver 动态修复聊天输入框样式
- 侧边栏标题、radio 按钮、metric 卡片等全部自定义样式

---

## 📁 文件结构

```
project2_knowledge_qa/
├── app.py                 # 主程序（含 UI 样式）
├── requirements.txt       # 依赖
├── .env                   # 环境变量（API Key）
├── knowledge_base/        # 知识库文档
│   ├── 产品介绍.txt
│   ├── 常见问题.txt
│   └── 公司介绍.txt
└── README.md             # 本文件
```

---

##  依赖安装

```bash
pip install -r requirements.txt
```

关键依赖：
- `streamlit` - Web 界面
- `transformers` - 语义模型（HuggingFace）
- `torch` - 模型推理
- `numpy` - 向量计算
- `httpx` - API 调用
- `python-dotenv` - 环境变量

---

##  相关链接

- DeepSeek API：https://platform.deepseek.com/
- HuggingFace Mirror：https://hf-mirror.com/
- 模型说明：https://huggingface.co/BAAI/bge-small-zh-v1.5
