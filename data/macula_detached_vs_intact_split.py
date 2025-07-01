import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def get_file_list(root, label, rel_dir):
    path = os.path.join(root, rel_dir)
    files = glob.glob(os.path.join(path, '*.mp4'))
    return [(f, label) for f in files]

def main():
    root = 'data/Dataset'
    data = []

    # Macula Detached
    data += get_file_list(root, 'Macula_Detached', 'Macula_Detached/Bilateral')
    data += get_file_list(root, 'Macula_Detached', 'Macula_Detached/TD')

    # Macula Intact
    data += get_file_list(root, 'Macula_Intact', 'Macula_Intact/ND')
    data += get_file_list(root, 'Macula_Intact', 'Macula_Intact/TD')

    # Create DataFrame
    df = pd.DataFrame(data, columns=['file_path', 'label'])

    # Shuffle and split
    train, temp = train_test_split(df, test_size=0.28, stratify=df['label'], random_state=42)
    val, test = train_test_split(temp, test_size=0.2, stratify=temp['label'], random_state=42)

    # Save
    train.to_csv('data/macula_detached_vs_intact_train.csv', index=False)
    val.to_csv('data/macula_detached_vs_intact_val.csv', index=False)
    test.to_csv('data/macula_detached_vs_intact_test.csv', index=False)

if __name__ == '__main__':
    main()
