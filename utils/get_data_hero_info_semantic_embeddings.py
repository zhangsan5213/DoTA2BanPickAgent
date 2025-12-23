import json
import torch
from sentence_transformers import SentenceTransformer

# 1. 准备数据 (这里根据你提供的 JSON 结构进行模拟)
with open('./data/hero_ability_descriptions.json', 'r', encoding='utf-8') as f:
    hero_data = json.load(f)

# 2. 加载模型 (请确保路径正确)
model_path = r'F:\Models\Qwen3-Embedding-0.6B'
model = SentenceTransformer(model_path)

# 3. 数据预处理：将每个英雄的技能 description 用 \n join
hero_names = []
hero_descriptions = []

for hero_name, skills in hero_data.items():
    # 获取该英雄所有技能的描述文字，并用换行符连接
    combined_skills = "\n".join(f'{skill_name}: {skill_desc}' for skill_name, skill_desc in skills.items())
    hero_names.append(hero_name)
    hero_descriptions.append(combined_skills)

# 4. 编码 (Encoding)
print(f"正在对 {len(hero_names)} 个英雄进行编码...")
# convert_to_tensor=True 会直接返回 torch.Tensor
embeddings = model.encode(hero_descriptions, convert_to_tensor=True)

# 5. 保存结果
# 我们通常保存一个字典，这样加载后可以通过英雄名索引到对应的向量
save_data = {
    hero_name: embedding
    for hero_name, embedding in zip(hero_names, embeddings)
}

torch.save(save_data, './data/hero_semantic_embeddings.pt')
print("编码完成并已保存至 ./data/hero_semantic_embeddings.pt")

# --- 验证加载 ---
loaded_data = torch.load('./data/hero_semantic_embeddings.pt')