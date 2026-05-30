"""
知识库问答系统 - 基于RAG架构 + 语义 Embedding
支持文档上传、向量检索、智能问答
"""

import os
import re

# HuggingFace 配置
os.environ["TQDM_DISABLE"] = "1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 本地开发时启用此镜像加速


os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

import streamlit as st
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


class SemanticModel:
    """语义模型封装类（使用 sentence-transformers）"""
    
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self.model = None
        
    def load(self):
        """加载模型"""
        # 修复：Streamlit 关闭 stdout，tqdm/transformers 调用 flush 时崩溃
        import sys
        if hasattr(sys.stdout, 'flush'):
            _real_flush = sys.stdout.flush
            def _safe_flush():
                try:
                    _real_flush()
                except (ValueError, OSError):
                    pass
            sys.stdout.flush = _safe_flush

        self.model = SentenceTransformer(self.model_name)
    
    def encode(self, texts):
        """生成语义向量"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")

# 页面配置
st.set_page_config(
    page_title="智能知识库",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式 - 浅灰绿配色主题
st.markdown("""
<style>
    /* ========== 全局 ========== */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }

    /* 主背景 - 浅灰色 */
    html, body,
    .stApp,
    .stApp > div,
    .stApp > div > div,
    .stApp > div > div > div,
    .stApp > div > div > div > div,
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlock"] > div,
    [data-testid="stVerticalBlock"] > div > div,
    .main,
    section.main,
    .main > div,
    .main > div > div {
        background: #EFEFEF !important;
        background-color: #EFEFEF !important;
    }

    [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }

    /* ========== 侧边栏 ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #315762 0%, #2A4D54 100%) !important;
    }

    /* 侧边栏所有内部容器透明 */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div > div {
        background: transparent !important;
    }

    /* 侧边栏标题、小标题透明背景 */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div,
    [data-testid="stSidebar"] .stMarkdown {
        background: transparent !important;
    }

    /* 侧边栏 metric 卡片透明 */
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: transparent !important;
    }

    /* 侧边栏 radio 按钮组透明 */
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] .stRadio > div > div {
        background: transparent !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        font-weight: 600 !important;
        color: #ffffff !important;
    }

    /* 侧边栏文字 - 白色高亮 */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.92) !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput > label,
    [data-testid="stSidebar"] .stTextInput > div > label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stTextInput span {
        color: rgba(255,255,255,0.85) !important;
    }

    /* ========== 侧边栏输入框 ========== */
    [data-testid="stSidebar"] .stTextInput > div {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] .stTextInput > div > div {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] input[data-testid="stTextInput"],
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] input[type="password"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
    }

    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: rgba(255,255,255,0.5) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.5) !important;
    }

    /* 侧边栏按钮 - 橙色系 */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #D0BA98 0%, #C4AA88 100%) !important;
        color: #23140C !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 20px !important;
        box-shadow: 0 4px 15px rgba(208, 186, 152, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(208, 186, 152, 0.4);
    }

    /* 侧边栏 Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
        margin: 14px 0 !important;
    }

    /* Metric卡片 */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 12px;
        padding: 12px !important;
        border: 1px solid rgba(255,255,255,0.2);
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #D0BA98 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }

    /* Radio 按钮组 */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        color: #ffffff !important;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.15) !important;
        border-color: rgba(208, 186, 152, 0.5) !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: rgba(208, 186, 152, 0.25) !important;
        border-color: #D0BA98 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label span,
    [data-testid="stSidebar"] .stRadio > div > label p {
        color: #ffffff !important;
    }

    /* Expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(255,255,255,0.03) !important;
    }

    [data-testid="stSidebar"] .streamlit-expanderContent p,
    [data-testid="stSidebar"] .streamlit-expanderContent span,
    [data-testid="stSidebar"] .streamlit-expanderContent caption {
        color: rgba(255,255,255,0.92) !important;
    }

    /* ========== 主内容区 ========== */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #23140C;
        margin-bottom: 0.5rem;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: rgba(35,20,12,0.6);
        margin-bottom: 1.5rem;
    }

    /* 文件预览区卡片 */
    .file-preview-card {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(208,186,152,0.3);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 8px 0 20px 0;
        max-height: 200px;
        overflow-y: auto;
    }

    .file-preview-title {
        color: #315762;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    .file-preview-content {
        background: rgba(35,20,12,0.05);
        border-radius: 10px;
        padding: 12px 14px;
        color: rgba(35,20,12,0.75);
        font-size: 0.82rem;
        line-height: 1.5;
        white-space: pre-wrap;
        max-height: 120px;
        overflow-y: auto;
    }

    .file-preview-meta {
        color: rgba(35,20,12,0.45);
        font-size: 0.74rem;
        margin-top: 8px;
    }

    /* 主区域分割线 */
    .main hr {
        border-color: rgba(208,186,152,0.3) !important;
    }

    /* 聊天消息样式 */
    .stChatMessage {
        background: rgba(255,255,255,0.6) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
        border: 1px solid rgba(208,186,152,0.2);
    }

    [data-testid="stChatMessage"] .stMarkdown {
        color: #23140C !important;
    }

    /* ========== 底部聊天输入区 ========== */
    .stBottom,
    .stBottom > div,
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] > div,
    [data-testid="stChatInputContainer"] > div > div,
    [data-testid="stChatInputContainer"] > div > div > div {
        background: #EFEFEF !important;
        background-color: #EFEFEF !important;
    }

    .stChatInput {
        background: #EFEFEF !important;
        padding: 24px 40px 32px 40px !important;
        border-top: 1px solid rgba(208,186,152,0.2) !important;
    }

    .stChatInput > div,
    .stChatInput > div > div,
    .stChatInput > div > div > div,
    .stChatInput > div > div > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 输入框 - 白色圆角 */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] > div > div > div > div {
        background: #ffffff !important;
        border-radius: 26px !important;
        padding: 8px 16px 8px 22px !important;
        max-width: 680px !important;
        margin: 0 auto !important;
        box-shadow: 0 4px 20px rgba(35,20,12,0.1) !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border: 1px solid rgba(208,186,152,0.3) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #23140C !important;
        font-size: 0.95rem !important;
        outline: none !important;
        flex: 1 !important;
        min-height: 22px !important;
        padding: 8px 4px !important;
        border-radius: 20px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(35,20,12,0.4) !important;
    }

    /* 发送按钮 */
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #315762 0%, #2A4D54 100%) !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        border: none !important;
        flex-shrink: 0 !important;
        color: #fff !important;
    }
</style>

<script>
// JS：强制修复聊天输入框
(function fixChatInput() {
    const tryFix = () => {
        const container = document.querySelector('[data-testid="stChatInput"]');
        if (!container) return false;
        
        const allDivs = container.querySelectorAll(':scope > div, :scope > div > div, :scope > div > div > div');
        
        allDivs.forEach(d => {
            d.style.background = 'transparent';
            d.style.border = 'none';
            d.style.boxShadow = 'none';
            d.style.padding = '0';
            d.style.margin = '0';
        });
        
        const textarea = container.querySelector('textarea');
        if (textarea) {
            let wrapper = textarea.parentElement;
            while (wrapper && wrapper !== container) {
                wrapper.style.background = '#ffffff';
                wrapper.style.borderRadius = '27px';
                wrapper.style.padding = '6px 12px 6px 24px';
                wrapper.style.maxWidth = '660px';
                wrapper.style.margin = '0 auto';
                wrapper.style.boxShadow = '0 4px 20px rgba(35,20,12,0.1)';
                wrapper.style.display = 'flex';
                wrapper.style.alignItems = 'center';
                wrapper.style.gap = '8px';
                wrapper.style.border = '1px solid rgba(208,186,152,0.3)';
                break;
            }
            
            textarea.style.background = 'transparent';
            textarea.style.border = 'none';
            textarea.style.boxShadow = 'none';
            textarea.style.color = '#23140C';
            textarea.style.outline = 'none';
            textarea.style.flex = '1';
            textarea.style.minHeight = '22px';
            textarea.style.padding = '8px 4px';
            textarea.style.borderRadius = '20px';
        }
        
        const btn = container.querySelector('button');
        if (btn) {
            btn.style.background = 'linear-gradient(135deg, #315762 0%, #2A4D54 100%)';
            btn.style.borderRadius = '50%';
            btn.style.width = '36px';
            btn.style.height = '36px';
            btn.style.border = 'none';
            btn.style.flexShrink = '0';
            btn.style.color = '#fff';
        }
        
        return true;
    };
    
    if (!tryFix()) {
        let attempts = 0;
        const interval = setInterval(() => {
            if (tryFix() || ++attempts > 20) clearInterval(interval);
        }, 200);
    }
    
    const observer = new MutationObserver(() => tryFix());
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# 补充样式
st.markdown("""
<style>
    /* 主区域按钮 */
    .main .stButton > button {
        background: rgba(49,87,98,0.1) !important;
        color: #315762 !important;
        border: 1px solid rgba(49,87,98,0.3) !important;
        border-radius: 12px !important;
    }

    .main .stButton > button:hover {
        background: rgba(49,87,98,0.15) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #315762 transparent transparent transparent !important;
    }

    /* Warning/Success */
    .stWarning {
        background: rgba(208,186,152,0.15) !important;
        border-radius: 12px !important;
        border: none !important;
        color: #8B7355 !important;
    }

    .stSuccess {
        background: rgba(49,87,98,0.1) !important;
        border-radius: 12px !important;
        border: none !important;
        color: #315762 !important;
    }

    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {display: none !important;}
</style>

<iframe id="style-frame" style="display:none;" srcdoc="
<style>
button {background-color: #ffffff !important; border-radius: 20px !important; border: 1px solid #315762 !important; color: #315762 !important;}
button * {color: #315762 !important;}
</style>
<script>
window.onload = function() {
    setTimeout(function() {
        var btns = document.querySelectorAll('[data-testid=stDeployButton]');
        btns.forEach(function(btn) {
            btn.style.cssText = 'background-color: #ffffff !important; border-radius: 20px !important; border: 1px solid #315762 !important;';
            btn.querySelectorAll('*').forEach(function(el) {
                el.style.color = '#315762 !important';
            });
        });
    }, 2000);
};
</script>
"></iframe>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "preview_file" not in st.session_state:
        st.session_state.preview_file = None


def load_documents():
    """加载知识库文档"""
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return []
    
    documents = []
    for file in os.listdir(KNOWLEDGE_DIR):
        if file.endswith('.txt'):
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'source': file
                    })
            except Exception as e:
                st.warning(f"加载 {file} 失败: {str(e)}")
    
    return documents


def split_documents(documents):
    """文档分块"""
    chunks = []
    for doc in documents:
        content = doc['content']
        lines = content.split('\n')
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if len(line) < 5:
                current_chunk.append(line)
                current_length += len(line)
                continue
            
            current_chunk.append(line)
            current_length += len(line)
            
            if current_length >= 100 or line.endswith(('。', '！', '？', '：')):
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > 20:
                    chunks.append({
                        'content': chunk_text,
                        'source': doc['source'],
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > 20:
                chunks.append({
                    'content': chunk_text,
                    'source': doc['source'],
                    'chunk_id': chunk_id
                })
    
    return chunks


def create_vectorstore(chunks, model):
    """创建语义向量库"""
    if not chunks or model is None:
        return None, None
    
    texts = [chunk['content'] for chunk in chunks]
    vectors = model.encode(texts)
    
    return model, vectors


def search_docs(query, model, vectors, chunks, top_k=5):
    """语义搜索"""
    if model is None or vectors is None:
        return []
    
    query_vec = model.encode([query])
    similarities = np.dot(query_vec, vectors.T).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.3:
            results.append({
                'content': chunks[idx]['content'],
                'source': chunks[idx]['source'],
                'score': float(similarities[idx])
            })
    
    return results


def ask_deepseek(context, question):
    """调用DeepSeek API回答问题"""
    import httpx
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""基于以下知识库内容回答问题。如果知识库中没有相关信息，请说明无法回答。

知识库内容：
{context}

问题：{question}

请给出简洁、准确的回答："""
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"调用API失败: {str(e)}"


def rebuild_index(model):
    """重建索引"""
    with st.spinner("正在加载文档..."):
        documents = load_documents()
    
    if not documents:
        return False, 0, None, None, []
    
    with st.spinner("正在分块..."):
        chunks = split_documents(documents)
    
    if not chunks:
        return False, 0, None, None, []
    
    with st.spinner("正在创建语义向量索引..."):
        model, vectors = create_vectorstore(chunks, model)
    
    return True, len(chunks), model, vectors, chunks


def main():
    init_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 💬 知识库助手")
        st.divider()
        
        api_key_input = st.text_input(
            "API Key", 
            value=DEEPSEEK_API_KEY, 
            type="password",
            placeholder="sk-xxxxx"
        )
        
        if api_key_input != DEEPSEEK_API_KEY:
            os.environ["DEEPSEEK_API_KEY"] = api_key_input
            st.session_state.embedding_model = None
            st.session_state.vectors = None
            st.session_state.chunks = []
        
        st.divider()
        st.markdown("**系统状态**")
        
        # 加载语义模型
        if st.session_state.embedding_model is None:
            with st.spinner("加载模型中..."):
                st.session_state.embedding_model = SemanticModel('BAAI/bge-small-zh-v1.5')
                st.session_state.embedding_model.load()
        
        col1, col2 = st.columns(2)
        with col1:
            status = "✅" if st.session_state.vectors is not None else "❌"
            st.metric("索引", status)
        with col2:
            st.metric("文档", st.session_state.doc_count)
        
        st.divider()
        st.markdown("**知识库文件**")
        files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith('.txt')] if os.path.exists(KNOWLEDGE_DIR) else []
        
        if files:
            # 添加"请选择..."选项作为默认
            radio_options = ["📂 请选择文件..."] + files
            current_index = 0
            if st.session_state.preview_file and st.session_state.preview_file in files:
                current_index = files.index(st.session_state.preview_file) + 1
            
            selected = st.radio(
                "📄 选择文件预览",
                radio_options,
                index=current_index
            )
            
            # 更新选中的文件（排除默认选项）
            if selected == "📂 请选择文件...":
                if st.session_state.preview_file is not None:
                    st.session_state.preview_file = None
                    st.rerun()
            elif selected != st.session_state.preview_file:
                st.session_state.preview_file = selected
                st.rerun()
        else:
            st.caption("暂无文件")
        
        st.divider()
        
        if st.button("🔄 重建索引", use_container_width=True):
            if not api_key_input:
                st.error("请输入 API Key")
            else:
                success, count, model, vectors, chunks = rebuild_index(st.session_state.embedding_model)
                if success:
                    st.session_state.embedding_model = model
                    st.session_state.vectors = vectors
                    st.session_state.chunks = chunks
                    st.session_state.doc_count = len(load_documents())
                    st.success(f"已加载 {count} 个文本块")
                else:
                    st.warning("知识库为空")
        
        st.divider()
        with st.expander("ℹ️ 使用说明"):
            st.caption("1. 输入 API Key")
            st.caption("2. 放入 TXT 文档")
            st.caption("3. 重建索引")
            st.caption("4. 开始问答")
    
    # 主内容区
    st.markdown('<p class="main-title">智能知识库问答</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">基于语义检索的 RAG 问答系统</p>', unsafe_allow_html=True)
    
    # 文件预览区 - 圆角卡片风格
    if st.session_state.preview_file:
        preview_path = os.path.join(KNOWLEDGE_DIR, st.session_state.preview_file)
        if os.path.exists(preview_path):
            with open(preview_path, 'r', encoding='utf-8') as pf:
                content = pf.read()
            st.markdown("""
            <div class="file-preview-card">
                <div class="file-preview-title">📄 {}</div>
                <div class="file-preview-content">{}</div>
                <div class="file-preview-meta">共 {} 字符</div>
            </div>
            """.format(st.session_state.preview_file, content[:2000] + ("..." if len(content) > 2000 else ""), len(content)), unsafe_allow_html=True)
    
    # 检查API密钥
    if not api_key_input:
        st.warning("⚠️ 请在侧边栏输入 DeepSeek API Key")
        st.stop()
    
    # 初始化向量库
    if st.session_state.vectors is None:
        documents = load_documents()
        if documents:
            with st.spinner("加载知识库..."):
                chunks = split_documents(documents)
                model, vectors = create_vectorstore(chunks, st.session_state.embedding_model)
                st.session_state.embedding_model = model
                st.session_state.vectors = vectors
                st.session_state.chunks = chunks
                st.session_state.doc_count = len(documents)
                st.success("知识库已加载")
    
    # 聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 输入区域
    if prompt := st.chat_input("输入问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.vectors is None:
                response = "请先重建索引"
            else:
                with st.spinner("检索中..."):
                    results = search_docs(
                        prompt, 
                        st.session_state.embedding_model,
                        st.session_state.vectors,
                        st.session_state.chunks
                    )
                
                if results:
                    context = "\n\n".join([r['content'] for r in results])
                    with st.spinner("生成回答..."):
                        response = ask_deepseek(context, prompt)
                else:
                    response = "未找到相关内容"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 底部按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.session_state.messages and st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
