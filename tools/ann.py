import os
import random
import pandas as pd

SUBTYPES = ['Well differentiated tubular adenocarcinoma',
            'Moderately differentiated tubular adenocarcinoma',
            'Poorly differentiated adenocarcinoma, non-solid type',
            'Poorly differentiated adenocarcinoma, solid type']
SUBTYPE_CLS = {
    SUBTYPES[0]: 0,
    SUBTYPES[1]: 1,
    SUBTYPES[2]: 2,
    SUBTYPES[3]: 2
}


def get_val(data_path='data/', val=0.8, seed=0):
    """Create val.txt, only once."""

    # read caption file
    data = pd.read_csv(os.path.join(data_path, 'captions.csv')).values.tolist()
    ids = {}
    id_i = [[] for _ in range(3)]
    for i, s, t in data:
        if s in SUBTYPE_CLS:
            ids[i] = SUBTYPE_CLS[s]
            id_i[SUBTYPE_CLS[s]].append(i)
    os.makedirs(f'data/gastric_cls3_ann', exist_ok=True)
    TRAIN_PATH = f'data/gastric_cls3_ann/train_all_{round(1 - val, 2)}.txt'
    VAL_PATH = f'data/gastric_cls3_ann/val_{val}.txt'
    print('Number of patches for each class:', [len(i) for i in id_i])

    # randomly sample train and val
    random.seed(seed)
    id_i = [random.sample(i, int(len(i) * val)) for i in id_i]

    file_i = [[] for _ in range(3)]
    file_list = os.listdir(os.path.join(data_path, 'patches_captions'))
    train_files = []
    n = [0, 0, 0]
    m = [0, 0, 0]
    for file in file_list:
        i = file.split('_')[0]
        for j, id_j in enumerate(id_i):
            if i in id_j:
                file_i[j].append(file + f' {j}')
                m[j] += 1
                break
        else:
            if i in ids:
                train_files.append(file + f' {ids[i]}')
                n[ids[i]] += 1
    val_files = sum(file_i, [])

    # write to file
    with open(TRAIN_PATH, 'w') as f:
        for file in train_files:
            f.write(f'{file}\n')
    with open(VAL_PATH, 'w') as f:
        for file in val_files:
            f.write(f'{file}\n')
    print(m, n)


def get_patient(train_file='data/gastric_cls3_ann/train_all_0.2.txt',
                num_patients=1,
                seed=0,
                sort=True):
    """Choose num_patients slides from each class, sorted by #patches."""

    # read train_all file
    with open(train_file, 'r') as f:
        train_files = [line[:-1] for line in f]
    file_i = [[] for i in range(3)]
    for file in train_files:
        cls = int(file.split(' ')[-1])
        file_i[cls].append(file)

    # sort by #patches
    random.seed(seed)
    if sort:
        id_i = dict()
        for i, fi in enumerate(file_i):
            count = dict()
            for file in fi:
                pid = file.split('_')[0]
                count[pid] = count.get(pid, []) + [file]
            count_list = [(pid, files) for pid, files in count.items()]
            count_list.sort(key=lambda x: len(x[1]), reverse=True)
            id_i[i] = sum([files for _, files in count_list[:num_patients]], [])
    else:
        id_i = {i: random.sample(fi, num_patients) for i, fi in enumerate(file_i)}

    for i in id_i:
        for j, file in enumerate(id_i[i]):
            id_i[i][j] = file.split('_')[0]
    train_file_i = [[] for i in range(3)]
    for i in id_i:
        for file in file_i[i]:
            if file.split('_')[0] in id_i[i]:
                train_file_i[i].append(file)
    train_files = sum(train_file_i, [])
    random.shuffle(train_files)

    # write to file
    save_file = train_file.replace('train_all', f'train_{num_patients}')
    with open(save_file, 'w') as f:
        f.write('\n'.join(train_files))


if __name__ == '__main__':
    get_val(val=0.8, seed=0)
    for i in [1, 2, 4, 8, 16]:
        get_patient(num_patients=i, sort=True)
