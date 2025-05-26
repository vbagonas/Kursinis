import pandas as pd
import random
from tqdm import tqdm

df = pd.read_csv('pair_train.csv')

if 'Code1_Length' not in df.columns:
    df['Code1_Length'] = df['Code1'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
if 'Code2_Length' not in df.columns:
    df['Code2_Length'] = df['Code2'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

filtered_df = df[
    (df['Code1_Length'] > 1) & (df['Code1_Length'] <= 530) & 
    (df['Code2_Length'] > 1) & (df['Code2_Length'] <= 530)
]

num_samples = 500

clone_df = filtered_df.sample(n=num_samples, random_state=42).copy()
clone_df['label'] = 1

nonclone_df = filtered_df.sample(n=num_samples, random_state=84).copy()

task_to_code2 = {}
for _, row in filtered_df.iterrows():
    task = row['Task']
    if task not in task_to_code2:
        task_to_code2[task] = []
    task_to_code2[task].append({
        'Code2': row['Code2'],
        'ID2': row.get('ID2', 'Unknown'),
        'Category2': row.get('Category2', 'Unknown')
    })

all_tasks = sorted(task_to_code2.keys())

nonclone_df['original_Task'] = nonclone_df['Task']
nonclone_df['original_Code2'] = nonclone_df['Code2']
nonclone_df['original_ID2'] = nonclone_df['ID2']
nonclone_df['replacement_Task'] = None
nonclone_df['replacement_ID2'] = None

max_task_distance = 5
for idx, row in tqdm(nonclone_df.iterrows(), total=len(nonclone_df)):
    current_task = row['Task']
    
    if current_task in all_tasks:
        current_idx = all_tasks.index(current_task)
        
        candidate_tasks = [task for i, task in enumerate(all_tasks) 
                          if abs(i - current_idx) > max_task_distance]
        
        if candidate_tasks and any(task_to_code2.get(task) for task in candidate_tasks):
            valid_tasks = [t for t in candidate_tasks if task_to_code2.get(t)]
            replacement_task = random.choice(valid_tasks)
            replacement_code = random.choice(task_to_code2[replacement_task])
            
            nonclone_df.at[idx, 'Code2'] = replacement_code['Code2']
            nonclone_df.at[idx, 'ID2'] = replacement_code['ID2']
            nonclone_df.at[idx, 'Category2'] = replacement_code['Category2']
            nonclone_df.at[idx, 'replacement_Task'] = replacement_task
            nonclone_df.at[idx, 'replacement_ID2'] = replacement_code['ID2']
    
    if pd.isna(nonclone_df.at[idx, 'replacement_Task']):
        other_tasks = [t for t in all_tasks if t != current_task and task_to_code2.get(t)]
        if other_tasks:
            replacement_task = random.choice(other_tasks)
            replacement_code = random.choice(task_to_code2[replacement_task])
            
            nonclone_df.at[idx, 'Code2'] = replacement_code['Code2']
            nonclone_df.at[idx, 'ID2'] = replacement_code['ID2']
            nonclone_df.at[idx, 'Category2'] = replacement_code['Category2']
            nonclone_df.at[idx, 'replacement_Task'] = replacement_task
            nonclone_df.at[idx, 'replacement_ID2'] = replacement_code['ID2']

nonclone_df['label'] = 0

combined_df = pd.concat([clone_df, nonclone_df])
combined_df = combined_df.sample(frac=1, random_state=123).reset_index(drop=True)

language_map = {
    'c': 'C',
    'java': 'Java',
    'py': 'Python',
    'cs': 'C#',
    'cpp': 'C++'
}

new_df = pd.DataFrame()
new_df['id'] = range(1, len(combined_df) + 1)
new_df['id1'] = combined_df['ID1']
new_df['id2'] = combined_df['ID2']
new_df['func1'] = combined_df['Code1']
new_df['func2'] = combined_df['Code2']
new_df['language1'] = combined_df['Category1'].map(language_map)
new_df['language2'] = combined_df['Category2'].map(language_map)
new_df['task1'] = combined_df['Task']
new_df['code1_length'] = combined_df['Code1_Length']
new_df['code2_length'] = combined_df['Code2_Length']
new_df['label'] = combined_df['label'].map({1: 'TRUE', 0: 'FALSE'})

replacement_task_map = {}
for _, row in combined_df.iterrows():
    key = (row['ID1'], row['ID2'])
    replacement_task_map[key] = row.get('replacement_Task', None)

new_df['task2'] = None
for i, row in new_df.iterrows():
    key = (row['id1'], row['id2'])
    
    if key in replacement_task_map and replacement_task_map[key] is not None:
        if row['label'] == 'FALSE':
            new_df.at[i, 'task2'] = replacement_task_map[key]
        else:
            new_df.at[i, 'task2'] = row['task1']
    else:
        new_df.at[i, 'task2'] = row['task1']

new_df.to_csv('clone_test_dataset.csv', index=False)
print(new_df.head())