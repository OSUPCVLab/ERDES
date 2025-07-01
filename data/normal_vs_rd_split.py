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

    # Normal
    data += get_file_list(root, 'Normal', 'Normal')

    # RD (all videos under these folders)
    rd_dirs = [
        'Macula_Detached/Bilateral',
        'Macula_Detached/TD',
        'Macula_Intact/ND',
        'Macula_Intact/TD'
    ]
    for d in rd_dirs:
        data += get_file_list(root, 'RD', d)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['file_path', 'label'])

    # Shuffle and split
    train, temp = train_test_split(df, test_size=0.28, stratify=df['label'], random_state=42)
    val, test = train_test_split(temp, test_size=0.2, stratify=temp['label'], random_state=42)

    # Save
    train.to_csv('data/normal_vs_rd_train.csv', index=False)
    val.to_csv('data/normal_vs_rd_val.csv', index=False)
    test.to_csv('data/normal_vs_rd_test.csv', index=False)

if __name__ == '__main__':
    main()
