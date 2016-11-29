# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:46:23 2016

@author: sergi

Idea: generate binary masks given the input train images

"""

import os
import numpy as np
import json
import cv2

def check_folder(name):
    if not os.path.isfile(name) and not os.path.isdir(name):
        os.mkdir(name)

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'SHARK', 'YFT'] #'OTHER' is missing: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation

current_class = classes[6]

# Read the json data
data = []
with open(current_class+'.json') as data_file:
    data = json.load(data_file)

# Choose the current image
#for img_idx in range(1): # (len(data)):
img_idx = 1 # Les anotacions estan be???
current_img_name = data[img_idx]['filename']
print('-- Processing image: '+current_img_name)

# Load the current image into a numpy array
img_path = os.path.join('../../input/train/',current_class,current_img_name)
img = cv2.imread(img_path)
img = np.array(img)

if len(data[img_idx]['annotations']) > 0:
    # Find the vector containing the head and tail, and generate the bounding box
    bbox = []
    head = np.array([0,0])
    tail = np.array([0,0])
    for idx_ann in range(0, len(data[img_idx]['annotations']), 2):
        head = np.array([data[img_idx]['annotations'][idx_ann]['x'],data[img_idx]['annotations'][idx_ann]['y']])
        tail = np.array([data[img_idx]['annotations'][idx_ann+1]['x'],data[img_idx]['annotations'][idx_ann+1]['y']])
        
        # Generate the bounding box
        x_min = int(np.min([head[0],tail[0]]))
        x_max = int(np.max([head[0],tail[0]]))
        y_min = int(np.min([head[1],tail[1]]))
        y_max = int(np.max([head[1],tail[1]]))
        curr_bbox = [x_min, x_max, y_min, y_max]
        
        # Add each bounding box
        bbox.append(curr_bbox)
        
    # Convert the bounding boxes to a numpy array
    bbox = np.array(bbox)
    
    # Generate the binary mask
    mask = np.zeros([img.shape[0],img.shape[1]])
    for idx_bb in range(len(bbox)):
        mask[bbox[idx_bb][2]:bbox[idx_bb][3],bbox[idx_bb][0]:bbox[idx_bb][1]] = 255
else:
    # Generate a black mask
    mask = np.zeros([img.shape[0],img.shape[1]])
    
# Write the masks to disk
output = '../../input/train_masks/'
check_folder(output)
output_dir = os.path.join(output,current_class)
check_folder(output_dir)
output_img = os.path.join(output_dir, current_img_name)
cv2.imwrite(output_img, mask)