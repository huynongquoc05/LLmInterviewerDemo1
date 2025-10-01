model_ids = [
    # "google/embeddinggemma-300M",  # bỏ qua nếu chưa hỗ trợ
    "dangvantuan/vietnamese-document-embedding",
    "AITeamVN/Vietnamese_Embedding",
    "keepitreal/vietnamese-sbert",
    "Alibaba-NLP/gte-multilingual-base",
    "hiieu/halong_embedding",
    "intfloat/multilingual-e5-large-instruct"
]

import torch
from sentence_transformers import SentenceTransformer

device = "cpu"

# --- Danh sách model cần trust_remote_code ---
models_need_trust = {
    "dangvantuan/vietnamese-document-embedding",
"Alibaba-NLP/gte-multilingual-base"
}

# Load model đầu tiên
first_model_id = model_ids[0]
if first_model_id in models_need_trust:
    model = SentenceTransformer(first_model_id, trust_remote_code=True).to(device=device)
else:
    model = SentenceTransformer(first_model_id).to(device=device)

print(f"Device: {model.device}")
print(model)
print("Total number of parameters in the model:", sum([p.numel() for _, p in model.named_parameters()]))

#####
words = ["apple thang", "banana", "car"]

embeddings = model.encode(words)
for idx, embedding in enumerate(embeddings):
    print(f"Embedding {idx+1} (shape): {embedding.shape}")

######
sentence_high = [
    "The chef created a wonderful dish to serve the diners.",
    "The cook made an exquisite dinner."
]

sentence_medium = [
    "She possesses strong knowledge of data science.",
    "He is passionate about studying deep learning."
]

sentence_low = [
    "It is raining lightly in Paris this morning.",
    "I plan to pick up some fresh vegetables from the store."
]

for sentence in [sentence_high, sentence_medium, sentence_low]:
    print("🙋‍♂️")
    print(sentence)
    embeddings = model.encode(sentence)

    similarities = model.similarity(embeddings[0], embeddings[1])
    print("`-> 🤖 score: ", similarities.numpy()[0][0])

###########
models = [model]
for model_id in model_ids[1:]:
    print(f"Load model {model_id} ....")
    if model_id in models_need_trust:
        m = SentenceTransformer(model_id, trust_remote_code=True).to(device=device)
    else:
        m = SentenceTransformer(model_id).to(device=device)
    models.append(m)

##########
sentence_high_vn = [
    "Ba con mèo ngồi xung quanh bốn con chó",
    "Bốn con chó ngồi giữa ba con mèo"
]
sentence_low_vn = [
    "Tôi rất thích ăn dưa hấu",
    "Trời Hà Nội hôm nay nóng"
]
sentence_very_low_vn = [
    "Tôi rất thích ăn dưa hấu",
    "Tôi không thích ăn dưa hấu"
]

for sentence in [sentence_high_vn, sentence_low_vn, sentence_very_low_vn]:
    print("🙋‍♂️")
    print(sentence)

    for i, m in enumerate(models):
        if i == 0:
            embeddings = m.encode(sentence)
        else:
            embeddings = m.encode(sentence)
        similarities = m.similarity(embeddings[0], embeddings[1])
        print(f"`-> 🤖 score of {model_ids[i]}: ", similarities.numpy()[0][0])

##############
import matplotlib.pyplot as plt
import numpy as np

def sim_for_pair(m, pair_sentences, use_prompt_sts=False):
    if use_prompt_sts:
        emb = m.encode(pair_sentences)
    else:
        emb = m.encode(pair_sentences)
    sim = m.similarity(emb[0], emb[1]).numpy()[0][0]
    return float(sim)

pairs = [
    ("High (nghĩa gần)", sentence_high_vn),
    ("Low (nghĩa xa)", sentence_low_vn),
    ("Very Low (trái nghĩa)", sentence_very_low_vn),
]

results = {}
for pair_name, pair_sents in pairs:
    results[pair_name] = []
    for i, m in enumerate(models):
        score = sim_for_pair(m, pair_sents, use_prompt_sts=(i == 0))
        results[pair_name].append(score)

for pair_name, scores in results.items():
    x = np.arange(len(model_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_ids)))

    plt.figure(figsize=(12, 8))
    plt.bar(x, scores, tick_label=model_ids, color=colors)

    plt.title(f"Similarity – {pair_name}")
    plt.ylabel("Similarity score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')

    for idx, val in enumerate(scores):
        plt.text(x[idx], val, f"{val:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
