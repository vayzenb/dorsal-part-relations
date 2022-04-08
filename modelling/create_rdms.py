curr_dir = '/home/vayzenbe/GitHub_Repos/docnet'

import sys
sys.path.insert(1, f'{curr_dir}/Models')
import os, argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps,  ImageFilter
from itertools import chain
import pandas as pd
import numpy as np

import cornet
import matplotlib.pyplot as plt
from statistics import mean
from LoadFrames import LoadFrames
from scipy import stats, spatial
import pdb

model_types = ['cornet_s']


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
euc = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

im_count =25


def extract_acts(model, im):
    """
    Extracts the activations for a series of images
    """
    model.eval()

    with torch.no_grad():

        im = im.cuda()
        output = model(im)
        output =output.view(output.size(0), -1)
        

    return output, label


def create_cos_rdm(output, label):
    """calculate cos similarity between each image"""
    rdm =[]
    im_pair = []
    for ii in range(0,len(output)):
        for kk in range(ii+1,len(output)):  \

            sim = cos(output[ii], output[kk])
            sim =sim.cpu().numpy()
            rdm.append(sim)
            im_pair.append([label[ii], label[kk]])

    return rdm, im_pair

def load_model(modelType_):


    if modelType_ == 'cornet_s':
        model = getattr(cornet, 'cornet_s')
        model = model(pretrained=False, map_location='gpu')
        checkpoint = torch.load('weights/cornet_s.pth')
        model.load_state_dict(checkpoint['state_dict'])
        layer = "avgpool"
        actNum = 512        

        decode_layer = nn.Sequential(*list(model.children())[0][4][:-3])
        model = nn.Sequential(*list(model.children())[0][:-1])
        model.add_module('4', decode_layer)
        im_dir = f'{curr_dir}/stim/original'

    elif modelType_ == 'skel':
        model = nn.Sequential(nn.Conv2d(3,1024,kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1),nn.ReLU(), nn.ConvTranspose2d(1024, 3, 224))
        checkpoint = torch.load('weights/skel_ae.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.Sequential(*list(model.children())[:-2])

        actNum = 1024
        im_dir = f'{curr_dir}/stim/blur'
    
    model = model.cuda()
    return model, im_dir

for mm in model_types:
    model, im_dir = load_model(mm)

    dataset = LoadFrames(im_dir,  transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=im_count,num_workers = 4, pin_memory=True)

    im, label = next(iter(loader))

    out, label = extract_acts(model,im)

    rdm, im_pair= create_cos_rdm(out, label)


    rdm = np.array(rdm)
    im_pair = np.array(im_pair)

    df = pd.DataFrame({'stim1': im_pair[:,0],'stim2' : im_pair[:,1],'similarity': rdm})
    df.to_csv(f'rdms/{mm}_rdm.csv', index=False)

