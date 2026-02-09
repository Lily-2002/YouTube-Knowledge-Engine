import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch.nn.functional as F

# --- 加载你的模型 ---
MODEL_PATH = "/home/fulian/RAG/Qwen_embed"
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name=MODEL_PATH, device=device, trust_remote_code=True)

def compare_sentences(sent_garbage, sent_useful):
    # 1. 计算两个句子的向量
    emb1 = torch.tensor(embed_model.get_text_embedding(sent_garbage)).to(device)
    emb2 = torch.tensor(embed_model.get_text_embedding(sent_useful)).to(device)
    
    # 2. 计算相似度 (Cosine Similarity)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    print(f"🗑️ 垃圾句: {sent_garbage}")
    print(f"💎 有用句: {sent_useful}")
    print(f"📊 相似度: {similarity:.4f}")
    
    # 模拟阈值测试
    threshold = 0.85  # 假设我们设定的删除线
    if similarity > threshold:
        print("⚠️ 危险！这两个句子太像了，可能会被误删！")
    else:
        print("✅ 安全。有用句子的差异足以让它逃过过滤器。")
    print("-" * 30)

# --- 测试案例 ---
# Case 1: 极度相似
compare_sentences("Can you hear the frequency?", "Now do you understand the frequency?")

# Case 2: 结构相似但内容不同
compare_sentences("Any questions?", "Any questions about the exam?")

# Case 3: 你的担心
compare_sentences("Let's get started.", "Let's get started with Python.")