# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:46:23 2016

@author: pepe
"""
import numpy as np
import json

from pprint import pprint

data = []
with open('shark_labels.json') as data_file:    
    data = json.load(data_file)


img_idx = 1
curent_img_name = data[img_idx]['filename']

v = np.array([0,0])
if len(data[0]['annotations']) > 0:
    for idx_ann in range(len(data[img_idx]['annotations'])):
        v[0] = data[img_idx]['annotations'][idx_ann]['x']
        v[1] = data[img_idx]['annotations'][idx_ann]['y']