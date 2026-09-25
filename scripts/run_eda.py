import pandas as pd
import os

print('--- DATASET AUDIT ---')

def audit_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.tsv')]
    for file in files:
        filepath = os.path.join(directory, file)
        try:
            df = pd.read_csv(filepath, sep='\t')
            print(f'\n[File]: {file}')
            print(f'  - Rows: {len(df)}')
            print(f'  - Columns: {list(df.columns)}')
            print(f'  - Missing values per column:')
            for col in df.columns:
                null_count = df[col].isnull().sum()
                print(f'      {col}: {null_count} ({null_count/len(df):.2%})')
            if 'entity_id' in df.columns:
                duplicates = df['entity_id'].duplicated().sum()
                print(f'  - Duplicate entity IDs: {duplicates}')
            if 'country' in df.columns:
                country_counts = df['country'].value_counts().to_dict()
                print(f'  - Countries: {country_counts}')
            
            if file == 'train_ground_truth.tsv' or file == 'test_ground_truth.tsv':
                if 'matched_entity_ids' in df.columns:
                    # Parse matches to get match counts
                    df['match_count'] = df['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) and x != '' else 0)
                    singletons = (df['match_count'] == 0).sum()
                    print(f'  - Singletons (match_count == 0): {singletons}')
                    print(f'  - Average matches per S1: {df["match_count"].mean():.2f}')
                    
        except Exception as e:
            print(f'  - Could not read {file}: {e}')

if os.path.exists('data/raw/train'):
    print('\n*** TRAIN ***')
    audit_files('data/raw/train')

if os.path.exists('data/raw/test'):
    print('\n*** TEST ***')
    audit_files('data/raw/test')
