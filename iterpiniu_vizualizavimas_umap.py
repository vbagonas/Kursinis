import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import ast

plt.rcParams['font.family'] = 'Calibri'

# Norint vizualizuoti kito kalbos modelio sugeneruotus įterpinius vietoj "codet5p_embeddings.csv" 
#       pakeisti į kito modelio įterpinių dokumentą: "sfr_embeddings.csv" arba "jina_embeddings.csv"
df = pd.read_csv("codet5p_embeddings.csv", sep=None, engine='python', on_bad_lines='warn')

all_embeddings = []
all_languages = []
all_tasks = []
all_lengths = []

for idx, row in df.iterrows():
    try:
        embedding_str = str(row['code1_embedding']).strip()
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            code1_emb = np.array(ast.literal_eval(embedding_str), dtype=np.float32)
        else:
            values = embedding_str.replace(',', ' ').split()
            code1_emb = np.array([float(v) for v in values if v], dtype=np.float32)
        
        all_embeddings.append(code1_emb)
        all_languages.append(row['language1'])
        all_tasks.append(str(row['task']))
        all_lengths.append(row['code1_length'])
    except:
        pass
    
    try:
        embedding_str = str(row['code2_embedding']).strip()
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            code2_emb = np.array(ast.literal_eval(embedding_str), dtype=np.float32)
        else:
            values = embedding_str.replace(',', ' ').split()
            code2_emb = np.array([float(v) for v in values if v], dtype=np.float32)
        
        all_embeddings.append(code2_emb)
        all_languages.append(row['language2'])
        all_tasks.append(str(row['task']))
        all_lengths.append(row['code2_length'])
    except:
        pass

embeddings = np.vstack(all_embeddings)

scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
reducer = umap.UMAP(n_components=2, random_state=48, n_neighbors=25, min_dist=0.1)
embedding_2d = reducer.fit_transform(embeddings_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(
    'CodeT5p kalbos modelio sugeneruotų įterpinių vizualizacija dvimatėje erdvėje', # Pakeitus modelio įterpinių duomenų dokumentą, šioje vietoje
    #                                                                                    reikia atitinkamai pakeisti pavadinimą
    fontsize=18,
    fontweight='bold',
    y=1
)

ax1 = axes[0]
lang_counts = {}
for lang in all_languages:
    lang_counts[lang] = lang_counts.get(lang, 0) + 1

sorted_languages = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
colors1 = plt.cm.Set1(np.linspace(0, 1, len(sorted_languages)))

for i, (lang, count) in enumerate(sorted_languages):
    mask = [l == lang for l in all_languages]
    ax1.scatter(embedding_2d[np.array(mask), 0], embedding_2d[np.array(mask), 1], 
               c=[colors1[i]], label=f'{lang}', alpha=0.7, s=80)

ax1.set_title('Pagal programavimo kalbą\n suskirstyti įterpiniai', fontsize=16)
ax1.set_xlabel('UMAP I-a dimensija', fontsize=14)
ax1.set_ylabel('UMAP II-a dimensija', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14, frameon=False)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
task_counts = {}
for task in all_tasks:
    task_counts[task] = task_counts.get(task, 0) + 1

top_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:10]
top_task_names = [task[0] for task in top_tasks]

other_mask = [t not in top_task_names for t in all_tasks]
if any(other_mask):
    ax2.scatter(embedding_2d[np.array(other_mask), 0], embedding_2d[np.array(other_mask), 1], 
               c='lightgray', label='Kitos užduotys', alpha=0.3, s=70)

colors2 = plt.cm.tab10(np.linspace(0, 1, len(top_tasks)))
for i, (task, count) in enumerate(top_tasks):
    mask = [t == task for t in all_tasks]
    ax2.scatter(embedding_2d[np.array(mask), 0], embedding_2d[np.array(mask), 1], 
               c=[colors2[i]], label=f'Užduotis nr. {task}', alpha=0.7, s=80)

ax2.set_title('Pagal 10 daugiausiai stebėjimų turėjusių\n užduočių suskirstyti įterpiniai', fontsize=16)
ax2.set_xlabel('UMAP I-a dimensija', fontsize=14)
ax2.set_ylabel('UMAP II-a dimensija', fontsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14, frameon=False)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()