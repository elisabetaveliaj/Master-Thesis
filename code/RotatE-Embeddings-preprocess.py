import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === SETTINGS ===
INPUT_CSV = 'DiaKG-triples.csv'
OUTPUT_DIR = 'DiaKG'

# === LOAD CSV ===
df = pd.read_csv(INPUT_CSV)

# === Sanitize all text entries ===
df['subject'] = df['subject'].str.replace(r'\s+', '_', regex=True)
df['predicate'] = df['predicate'].str.replace(r'\s+', '_', regex=True)
df['object'] = df['object'].str.replace(r'\s+', '_', regex=True)

# === Collect Entities and Relations ===
entities = sorted(set(df['subject']).union(df['object']))
relations = sorted(set(df['predicate']))

entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {rel: idx for idx, rel in enumerate(relations)}

# === Split the data: 70% train, 10% valid, 20% test ===
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(train_val_df, test_size=0.125, random_state=42) 

# === Make Output Directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Write train.txt, valid.txt, test.txt using tab separator ===
train_df.to_csv(f'{OUTPUT_DIR}/train.txt', sep='\t', index=False, header=False)
valid_df.to_csv(f'{OUTPUT_DIR}/valid.txt', sep='\t', index=False, header=False)
test_df.to_csv(f'{OUTPUT_DIR}/test.txt', sep='\t', index=False, header=False)

# === Write entities.dict with tab separator ===
with open(f'{OUTPUT_DIR}/entities.dict', 'w', encoding='utf-8') as f:
    for entity, idx in entity2id.items():
        f.write(f"{idx}\t{entity}\n")

# === Write relations.dict with tab separator ===
with open(f'{OUTPUT_DIR}/relations.dict', 'w', encoding='utf-8') as f:
    for rel, idx in relation2id.items():
        f.write(f"{idx}\t{rel}\n")

print(f"Done. Files written to: {OUTPUT_DIR}")
