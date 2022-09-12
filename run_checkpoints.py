#script just to enable a subjective eval of input data using a predeterined checkpoint
import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl

from train import XUMXManager

parser = argparse.ArgumentParser()

import yaml

#paths for config files:
serialized_model_path = os.path.join('exp_outputs' , 'serialized_model')
train_info_data_path = os.path.join('exp_outputs' , 'train_data_info_dict')
epoch_path = os.path.join('exp_outputs', 'epoch=7-step=36575.ckpt')



state_dict = torch.load(epoch_path, map_location=torch.device('cpu'))
model = torch.load(serialized_model_path, map_location=torch.device('cpu'))
train_info = torch.load(train_info_data_path, map_location=torch.device('cpu'))

#import pdb
#pdb.set_trace()

modified_state_dict = {(key[6:] if key[0:5] == 'model' else key) : val 
                        for key, val in state_dict['state_dict'].items()}

model['state_dict'] = modified_state_dict
model.update(train_info)

model['state_dict'].pop('loss_func.transform.0.window', None)
torch.save(model, "test.pth")


