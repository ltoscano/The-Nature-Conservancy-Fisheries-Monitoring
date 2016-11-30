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

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'SHARK', 'YFT', 'OTHERS']
iinput = '../../input/train/'
output = '../../input/train_masks/'
img_h = 720
img_w = 1280

for current_class in classes:
    print('-- Processing folder: '+current_class)
    
    if current_class == 'NoF':
        # No fishes there 
    else:
        # Read the json data
        data = []
        with open(current_class+'.json') as data_file:
            data = json.load(data_file)
        
        # Choose the current image
        for img_idx in range(len(data)):
            current_img_name = data[img_idx]['filename']
            if current_class == 'SHARK' or current_class == 'OTHERS' or current_class == 'YFT':
                current_img_name = current_img_name.rsplit('/', 1)[1]
            print('-- Processing image: '+current_img_name)
            
            # Load the current image into a numpy array
            img_path = os.path.join(iinput,current_class,current_img_name)
            img = cv2.imread(img_path)
            img = np.array(img)
            
            if len(data[img_idx]['annotations']) > 0:
                bbox = []
                head = np.array([0,0])
                tail = np.array([0,0])
                for idx_ann in range(len(data[img_idx]['annotations'])):
                    # Extract the data from the json files
                    x_min = int(np.max(data[img_idx]['annotations'][idx_ann]['x'],0))
                    y_min = int(np.max(data[img_idx]['annotations'][idx_ann]['y'],0))
                    width = int(np.max(data[img_idx]['annotations'][idx_ann]['width'],0))
                    height = int(np.max(data[img_idx]['annotations'][idx_ann]['height'],0))
                    x_max = int(x_min + width)
                    y_max = int(y_min + height)
                    curr_bbox = [x_min, x_max, y_min, y_max]
                    
                    # Add each bounding box
                    bbox.append(curr_bbox)
                    
                # Convert the bounding boxes to a numpy array
                bbox = np.array(bbox)
                
                # Generate the binary mask
                mask = np.zeros([img_h,img_w])
                for idx_bb in range(len(bbox)):
                    mask[bbox[idx_bb][2]:bbox[idx_bb][3],bbox[idx_bb][0]:bbox[idx_bb][1]] = 255
            else:
                # Generate a black mask
                mask = np.zeros([img_h,img_w])
            
            # Write the masks to disk
            check_folder(output)
            output_dir = os.path.join(output,current_class)
            check_folder(output_dir)
            output_img = os.path.join(output_dir, current_img_name)
            
            #To check if the mask is correct do mask/255*img[:,:,0]
            cv2.imwrite(output_img, mask)