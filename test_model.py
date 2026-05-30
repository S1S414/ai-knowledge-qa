"""测试模型加载"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

print("Testing transformers import...")
from transformers import AutoTokenizer, AutoModel

print("Loading model BAAI/bge-small-zh-v1.5...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
print("Success!")
