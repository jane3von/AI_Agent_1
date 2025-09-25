import os
from modelscope import snapshot_download

# 在运行代码前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
# 或者
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='./model_weights', revision='master')
# model_dir = snapshot_download('QWen/Qwen2.5-0.5B-Instruct', cache_dir='.', revision='master')
# model_dir = snapshot_download('QWen/Qwen2.5-1.5B-Instruct', cache_dir='./model_weights', revision='master')

# model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir='./model_weights', revision='master')
# model_dir = snapshot_download('Qwen/Qwen3-1.7B', cache_dir='./model_weights', revision='master')
#model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='./model_weights', revision='master')

#model_dir = snapshot_download('Qwen/Qwen1.5-1.8B', cache_dir='.', revision='master')
 
#model_dir = snapshot_download('Qwen/Qwen-7B', cache_dir='.', revision='master')

#model_dir = snapshot_download('BAAI/bge-base-zh', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

#model_dir = snapshot_download('QWen/Qwen2.5-0.5B-Instruct', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

#model_dir = snapshot_download('BAAI/bge-m3', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

# BAAI/bge-reranker-v2-m3
#model_dir = snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

# Qwen-VL-7B
#model_dir = snapshot_download('Qwen/Qwen2-VL-7B-Instruct', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

# Salesforce/blip2-opt-2.7b
#model_dir = snapshot_download('Salesforce/blip2-opt-2.7b', cache_dir=r'C:\Users\zhang\Documents\models', revision='master')

# /Users/pc/Documents/model/Qwen
#model_dir = snapshot_download('QWen/Qwen2.5-0.5B-Instruct', cache_dir=r'/Users/pc/Documents/models/Qwen', revision='master')

#model_dir = snapshot_download('damo/nlp_corom_sentence-embedding_chinese-base', cache_dir=r'/Users/pc/Documents/models', revision='master')

# 轻量级嵌入模型
#model_dir = snapshot_download('sentence-transformers/all-MiniLM-L6-v2', cache_dir=r'/Users/pc/Documents/models', revision='master')

model_dir = snapshot_download('BAAI/bge-reranker-base', cache_dir=r'/Users/pc/Documents/models', revision='master')