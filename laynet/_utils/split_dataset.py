raise NotImplementedError('script needs further editing to be run! i.a. move to top level')

import random
import os
import shutil

def split(root_dir, 
        test=0.2, 
        val=0.17,
        reshuffle=False,
        data_str='_rgb.nii.gz',
        label_str='_l.nii.gz'):

    assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
    'root_dir not a valid directory'

    assert test+val<1, 'Invalid input'
    
    
    assert isinstance(data_str, str) and isinstance(label_str, str), \
    'data_str or label_str not valid.'
    
    # get all files in root_dir
    all_files = os.listdir(path = root_dir)
    # extract the data files
    data = [el for el in all_files if el[-len(data_str):] == data_str]
    
    assert len(data) == \
        len([el for el in all_files if el[-len(label_str):] == \
        label_str]), \
        'Amount of data and label files not equal.'

    if len(data) == 0:
        print('No data in root_dir found. Already split')
    else:
        # split the dataset in train, test, val
        
        # create paths
        train_dir = os.path.join(root_dir, 'train')
        test_dir = os.path.join(root_dir, '.test')
        val_dir = os.path.join(root_dir, 'val')
        assert os.path.exists(train_dir), 'Please mkdir'
        assert os.path.exists(test_dir), 'Please mkdir'
        assert os.path.exists(val_dir), 'Please mkdir'
        
        # sort and shuffle
        data.sort()
        if not reshuffle:
            random.seed(1220)
        random.shuffle(data)

        spl1 = int(val * len(data))
        spl2 = int(test * len(data))
        val_data = data[:spl1]
        test_data = data[spl1:spl1+spl2]
        train_data = data[spl1+spl2:]
        
        # debug
        assert val_data[-1] != test_data[0]
        assert test_data[-1] != train_data[0]

        train_data.sort()
        test_data.sort()
        val_data.sort()
        print('Splitting dataset.. Size', len(data))
        print('Train: ', len(train_data))
        print('Test:  ', len(test_data))
        print('Val:   ', len(val_data))

        for cdata, cdir in (
                (train_data, train_dir),
                (test_data, test_dir),
                (val_data, val_dir)):
            # print(cdata, cdir)
            for file in cdata:
                data_src = os.path.join(root_dir, file)
                # data_dest = os.path.join(cdir, file)

                label_file = file.replace(data_str, label_str)
                label_src = os.path.join(root_dir, label_file)
                # label_dest = os.path.join(cdir, label_file)
                
                # print(data_src, data_dest)
                # print(label_src, label_dest)
                shutil.move(data_src, cdir)
                shutil.move(label_src, cdir)

def merge(root_dir):
    # create paths
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    val_dir = os.path.join(root_dir, 'val')
    assert os.path.exists(train_dir), 'Dataset is not split'
    assert os.path.exists(test_dir), 'Dataset is not split'
    assert os.path.exists(val_dir), 'Dataset is not split'
    
    print('merging dataset..')

    for cdir in (train_dir, test_dir, val_dir):
        for file in os.listdir(cdir):
            # print(os.path.join(cdir, file), root_dir)

            shutil.move(os.path.join(cdir, file), root_dir)

ddir = '/home/stefan/data/fullDataset/labeled_prep3'
split(ddir)
# merge(ddir)
