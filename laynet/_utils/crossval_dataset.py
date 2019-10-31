
raise NotImplementedError('script needs further editing to be run! i.a. move to top level')

import os
import random
import shutil
# training set size 20, val set size 5

# original dataset is in layer_seg
# rotate dataset to perform 5x cross validation

data_str='_rgb.nii.gz'
label_str='_l.nii.gz'


root_dir = '/home/gerlstefan/data/fullDataset/labeled'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')


assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
root_dir + ' not a valid directory'

assert os.path.exists(train_dir) and os.path.isdir(train_dir), \
train_dir + ' not a valid directory'

assert os.path.exists(val_dir) and os.path.isdir(val_dir), \
val_dir + ' not a valid directory'

assert isinstance(data_str, str) and isinstance(label_str, str), \
'data_str or label_str not valid.'

# get all files in train_dir
train_files = os.listdir(path = train_dir)
# extract the data files
train_data = [el for el in train_files if el[-len(data_str):] == data_str]

assert len(train_data) == \
    len([el for el in train_files if el[-len(label_str):] == \
    label_str]), \
    'Amount of data and label files not equal.'

if len(train_data) == 0:
    print('No data in ' + train_dir + ' found. Already split')

# get all files in val_dir
val_files = os.listdir(path = val_dir)
# extract the data files
val_data = [el for el in val_files if el[-len(data_str):] == data_str]

assert len(val_data) == \
    len([el for el in val_files if el[-len(label_str):] == \
    label_str]), \
    'Amount of data and label files not equal.'

if len(val_data) == 0:
    print('No data in' + val_dir +' found. Already split')


# now, the data in train_dir needs to be shuffled.
train_data.sort()
random.seed(314)
random.shuffle(train_data)

# now perform a split in 4 x 5 samples

train_data_split = [train_data[0:5]]
train_data_split.append(train_data[5:10])
train_data_split.append(train_data[10:15])
train_data_split.append(train_data[15:])

assert len(train_data_split[0]) == len(train_data_split[1])
assert len(train_data_split[1]) == len(train_data_split[2])
assert len(train_data_split[2]) == len(train_data_split[3])

# generate 5 new datasets

dest_dir = '/home/gerlstefan/data/fullDataset/crossval'

for rot in range(5):

    dest_train = os.path.join(dest_dir, str(rot), 'train')
    dest_val = os.path.join(dest_dir, str(rot), 'val')

    if not os.path.exists(dest_train):
        os.makedirs(dest_train)
        print('mkdir', dest_train)

    if not os.path.exists(dest_val):
        os.makedirs(dest_val)
        print('mkdir', dest_val)


    # assign correct lists to 
    # dest_train_data
    # dest_val_data
    
    # this is the list of all data,
    # with all_data[0] is the initial val data
    # and all_data[1:] is the initial training data
    all_data = ([val_data] + train_data_split).copy()

    dest_val_data = all_data[rot]
    all_data.pop(rot)
    dest_train_data = [j for i in all_data for j in i]

    # copy the files
    # TRAIN
    for file in dest_train_data:
        if not os.path.exists(os.path.join(train_dir, file)):
            dirr = val_dir
        else:
            dirr = train_dir
        
        data_src = os.path.join(dirr, file)
        label_file = file.replace(data_str, label_str)

        label_src = os.path.join(dirr, label_file)

        # data_dest = os.path.join(dest_train, file)
        shutil.copy(data_src, dest_train)
        shutil.copy(label_src, dest_train)
    print('SPLIT', rot)
    print('TRAIN DATA:')
    print(*dest_train_data, sep='\n')
    print('VAL DATA:')
    print(*dest_val_data, sep='\n')

    # VAL
    for file in dest_val_data:
        if not os.path.exists(os.path.join(train_dir, file)):
            dirr = val_dir
        else:
            dirr = train_dir
        data_src = os.path.join(dirr, file)
        label_file = file.replace(data_str, label_str)

        label_src = os.path.join(dirr, label_file)

        # data_dest = os.path.join(dest_train, file)
        shutil.copy(data_src, dest_val)
        shutil.copy(label_src, dest_val)





