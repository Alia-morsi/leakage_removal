import pandas as pd
import os
import numpy as np

from room_simulator import to_room_dict, room_dicts_equal

dataroot = '/media/data/alia/Documents/datasets/leakage_removal2/'
instrument = 'drums'

#set1 and set2 are dataframes
def compare_sets(set1, set2):
    set1_array = []
    set2_array = []
    for index, row in set1.iterrows():
        set1_array.append(to_room_dict(row))
    
    for index, row in set2.iterrows():
        set2_array.append(to_room_dict(row))

    matches = []
    for set1_obj in set1_array:
        for set2_obj in set2_array:
            equal, diff = room_dicts_equal(set1_obj, set2_obj)
            if equal:
                matches.append((set1_obj, set2_obj, diff))

    return matches

if __name__ == '__main__':
    train_path = os.path.join(dataroot, 'train', instrument, 'room_params.csv')
    test_path = os.path.join(dataroot, 'test', instrument, 'room_params.csv')

    #compare train and test
    train_params_df = pd.read_csv(train_path)
    test_params_df = pd.read_csv(test_path)

    matches = compare_sets(train_params_df, test_params_df)
    import pdb
    pdb.set_trace()
    
