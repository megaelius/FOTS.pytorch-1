import os
import cv2
import copy
import time
import tqdm
import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from pathlib import Path

from tools.models import ModelResNetSep2, OwnModel
import tools.net_utils as net_utils
from src.utils import strLabelConverter, alphabet
from evaluation import recognize_plate, PlateRNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model', default='./weights/FOTS_280000.h5')
    parser.add_argument('--plate_model', default='../out/Model_Bidi')

    # parser.add_argument('-model', default='./weights/e2e-mlt.h5')
    parser.add_argument('--segm_thresh', type=float,default=0.5)
    parser.add_argument('--input')
    parser.add_argument('--output')

    font2 = ImageFont.truetype("./tools/Arial-Unicode-Regular.ttf", 18)

    args = parser.parse_args()

    converter = strLabelConverter(alphabet)

    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Using cuda ...')
        else:
            device = torch.device('cpu')
            print('No cuda GPU available, defaulting to CPU ...')
    else:
        device = torch.device('cpu')

    net = ModelResNetSep2(attention=True, nclass=len(alphabet)+1)
    net_utils.load_net(args.model, net, device)
    net = net.eval()
    net.to(device)

    #Load plateset
    with open('../../Data/PlateSet.pkl', 'rb') as f:
        plateset = pickle.load(f)

    #print('N818LS' in plateset)
    #print('N818LS' in confusable_plates('NB1BLS'))

    #Load char to index dictionary
    with open(os.path.join(args.plate_model,'char_to_idx.pkl'), 'rb') as handle:
        char_to_idx = pickle.load(handle)

    #platenet = PlateNet(num_embeddings = len(char_to_idx) + 1, embedding_dim = 2)
    platenet = torch.load(os.path.join(args.plate_model,'weights.pt'), map_location = device)
    platenet.eval()
    platenet.to(device)

    im = cv2.imread(args.input)
    plate, confidence = recognize_plate(im,net,args.segm_thresh,platenet,plateset,converter,font2,char_to_idx,device,args.input,args.output,debug = True)
