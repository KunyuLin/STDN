import os

ROOT_DATASET = './'  

def return_ucfhmdb(modality):
    filename_categories = 12
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'data/'
        filename_imglist_train = 'datalists/UCF-HMDB-DG-list/ucf_train.txt'
        filename_imglist_val = 'datalists/UCF-HMDB-DG-list/ucf_val.txt'
        filename_imglist_test = 'datalists/UCF-HMDB-DG-list/hmdb_full.txt'
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data, prefix


def return_hmdbucf(modality):
    filename_categories = 12
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'data/'
        filename_imglist_train = 'datalists/UCF-HMDB-DG-list/hmdb_train.txt'
        filename_imglist_val = 'datalists/UCF-HMDB-DG-list/hmdb_val.txt'
        filename_imglist_test = 'datalists/UCF-HMDB-DG-list/ucf_full.txt'
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {
        'ucfhmdb': return_ucfhmdb, 
        'hmdbucf': return_hmdbucf
    }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, file_imglist_test, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_imglist_test = os.path.join(ROOT_DATASET, file_imglist_test)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, file_imglist_test, root_data, prefix

