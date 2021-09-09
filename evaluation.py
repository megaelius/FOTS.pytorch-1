'''
Created on 2019-06-30

@author: chenjun2hao
'''

import os
import cv2
import copy
import glob
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
from unicodedata import normalize
from torch.nn.utils.rnn import pad_sequence
from collections import deque

from nms import get_boxes

from tools.models import ModelResNetSep2, OwnModel
import tools.net_utils as net_utils
from src.utils import strLabelConverter, alphabet
from tools.ocr_utils import ocr_image, align_ocr
from tools.data_gen import draw_box_points

def resize_image(im, max_size = 1585152, scale_up=True):

    if scale_up:
        image_size = [im.shape[1] * 3 // 32 * 32, im.shape[0] * 3 // 32 * 32]
    else:
        image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
    while image_size[0] * image_size[1] > max_size:
        image_size[0] /= 1.2
        image_size[1] /= 1.2
        image_size[0] = int(image_size[0] // 32) * 32
        image_size[1] = int(image_size[1] // 32) * 32

    resize_h = int(image_size[1])
    resize_w = int(image_size[0])

    scaled = cv2.resize(im, dsize=(resize_w, resize_h))
    return scaled, (resize_h, resize_w)

def is_plate(s):
    '''
    Returns True if the string s follows the format of an aircraft registration.
    '''
    df = pd.read_csv('./Aircraft_registration_prefixes.csv',sep=';')
    #prefixes = [p.replace('-','') for p in df['Regn_Prefix']]
    prefixes = list(df['Regn_Prefix']) + list(df['Old_Prefix'])
    if len(s) >= 4:
        return s[:1] in prefixes \
            or s[:2] in prefixes \
            or s[:3] in prefixes \
            or s[:4] in prefixes
    else:
        return False

def normalize2(s,form: str):
    return normalize(form,s)

def frame_from_video(video):
    i = 0
    while video.isOpened():
        success, frame = video.read()
        if not i % 16:
            if success:
                yield frame
            else:
                break
        i+=1

def index_chars(word,char_to_idx):
    result = []
    for char in word:
        if char in char_to_idx:
            result.append(char_to_idx[char])
        else:
            result.append(0)
    return result

class PlateNet(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = torch.nn.Linear(embedding_dim, 1, bias=False)
    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input, input_lengths):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)
        u = e.sum(dim=1)
        # u shape is (B, E)
        v = self.lin(u)
        # v shape is (B, V)
        return v

class PlateRNN(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
        if bidirectional:
            self.h2o = torch.nn.Linear(2*hidden_size, output_size)
        else:
            self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, input_lengths):
        # T x B
        encoded = self.embed(input)
        # T x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths, batch_first = True)
        # Packed T x B x E
        output, _ = self.rnn(packed)
        # Packed T x B x E

        # Important: you may need to replace '-inf' with the default zero padding for other pooling layers
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True)
        # T x B x H
        output = padded.sum(dim=1)
        # B x H
        output = self.h2o(output)
        # B x O
        return output

def confusable_plates(plate):
    confusions = [['3','8','B'],
                  ['B','R','8'],
                  ['I','1','T','L'],
                  ['7','T'],
                  ['A','4'],
                  ['0','O','Q','D'],
                  ['Z','2','7'],
                  ['G','6'],
                  ['G','C'],
                  ['M','N'],
                  ['Y','V'],
                  ['V','W'],
                  ['U','J'],
                  ['K','X'],
                  ['E','F'],
                  ['5','S','9']]
    #Create confusion map with every character as key and
    #its possible confussions as values
    conf_map = {}
    for conf in confusions:
        for i in range(len(conf)):
            values = []
            key = conf[i]
            for j in range(len(conf)):
                if j != i:
                    values.append(conf[j])
            if key in conf_map:
                conf_map[key] += values
            else:
                conf_map[key] = values
    '''
    print(conf_map)
    for c in conf_map:
        s = f'{c} &'
        for ch in conf_map[c]:
            s+=f' {ch},'
        s=s[:-1]
        s+=' \\\\'
        print(s)
    '''
    #initiallize list with empty character
    n = 1
    out = [('',0)]
    for i in range(len(plate)):
        newout = []
        conf = [plate[i]]
        if plate[i] in conf_map:
            conf += conf_map[plate[i]]
        n*=len(conf)
        for char in conf:
            for w,edits in out:
                newout.append((w+char,edits + (char!=plate[i])))
        out = newout
    return dict(out)




    '''
    #Iterate over characters, keep a list of candidates to modify them at the
    #actual position and add new candidates
    out = {plate:0}
    candidates = deque()
    candidates.append((plate,0))
    i = 0
    while i < len(plate):
        if plate[i] in conf_map:
            while len(candidates):
                p,edits = candidates.pop()
                char = p[i]
                newcand = []
                newcand.append((p,edits))
                for cp in conf_map[char]:
                    pp = list(p)
                    print(p)
                    pp[i] = cp
                    pp = ''.join(pp)
                    print(pp)
                    out[pp] = edits+1
                    newcand.append((pp,edits+1))
            for cand in newcand:
                candidates.append(cand)
        i += 1
    '''
    return out


def recognize_plate(im,net,segm_thresh,platenet,plateset,converter,font2,char_to_idx,device,path=None,output_folder=None,debug=False):
    times = {'FOTS':0,'Classifier':0,'Refinement':0,'Detections':0}
    im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
    images = np.asarray([im_resized], dtype=float)
    images /= 128
    images -= 1
    im_data = net_utils.np_to_variable(images.transpose(0, 3, 1, 2))
    im_data = im_data.to(device)

    init_FOTS = time.time()
    seg_pred, rboxs, angle_pred, features = net(im_data)

    rbox = rboxs[0].data.cpu()[0].numpy()           # 转变成h,w,c
    rbox = rbox.swapaxes(0, 1)
    rbox = rbox.swapaxes(1, 2)

    angle_pred = angle_pred[0].data.cpu()[0].numpy()

    segm = seg_pred[0].data.cpu()[0].numpy()
    segm = segm.squeeze(0)

    boxes =  get_boxes(segm, rbox, angle_pred, segm_thresh)

    if path is not None and output_folder is not None:
        draw2 = np.copy(im_resized)
        img = Image.fromarray(draw2)
        draw = ImageDraw.Draw(img)

    out_boxes = []
    texts = []
    plate = None
    confidence = 0

    for box in boxes:

        pts  = box[0:8]
        pts = pts.reshape(4, -1)

        # det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
        det_text, conf, dec_s = align_ocr(net, converter, im_data, box, features,device, debug=0)
        if len(det_text) == 0:
          continue
        texts.append(det_text)
        if path is not None and output_folder is not None:
            width, height = draw.textsize(det_text, font=font2)
            center =  [box[0], box[1]]
            draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
        out_boxes.append(box)
        #print(det_text, conf, dec_s)
        '''
        if is_plate(det_text) and conf > confidence:
            plate = det_text
            confidence = conf
        '''
    finish_FOTS = time.time()

    if debug:
        print(texts)

    init_PlateClassifier = time.time()
    with torch.no_grad():
        texts_idx = [torch.tensor(index_chars(w,char_to_idx)) for w in texts]
        if len(texts_idx) > 0:
            sequences= pad_sequence(texts_idx, batch_first = True).to(device)
            input_lengths = [len(s) for s in sequences]
            output = platenet(sequences,input_lengths)
            val, idx = torch.max(torch.nn.Sigmoid()(output),dim=0)
            plate = texts[idx]
            confidence = val
            if debug:
                print(torch.nn.Sigmoid()(output))
    finish_PlateClassifier = time.time()

    '''
    Recognition refinement
    '''
    init_RR = time.time()
    if plate is not None:
        if plate not in plateset:
            plates_and_edits = confusable_plates(plate)
            if debug:
                print(plates_and_edits)
            possible_plates = set(plates_and_edits.keys())
            plates = possible_plates.intersection(plateset)
            if len(plates) > 0:
                if debug:
                    for p in plates:
                        print(f'{p}: {plates_and_edits[p]}')
                plate = None
                min = 100
                for p in plates:
                    edits = plates_and_edits[p]
                    if edits < min:
                        plate = p
                        min = edits
            else:
                plate = None
                confidence = 0
        if debug:
            print(plate)
    finish_RR = time.time()

    times['FOTS'] = finish_FOTS-init_FOTS
    times['Classifier'] = finish_PlateClassifier-init_PlateClassifier
    times['Refinement'] = finish_RR-init_RR
    times['Detections'] = len(texts)

    if path is not None and output_folder is not None:
        im = np.array(img)
        for box in out_boxes:
            pts  = box[0:8]
            pts = pts.reshape(4, -1)
            draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

        #cv2.imshow('img', im)
        if not os.path.isdir(os.path.join(output_folder, path.split('/')[-3])):
            Path(os.path.join(output_folder, path.split('/')[-3])).mkdir()
        out_filename = os.path.join(output_folder, path.split('/')[-3], os.path.basename(path))
        cv2.imwrite(out_filename, im)

    return plate, float(confidence), times

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model', default='./weights/FOTS_280000.h5')
    parser.add_argument('--plate_model', default='../out/Model_Augmentation')

    # parser.add_argument('-model', default='./weights/e2e-mlt.h5')
    parser.add_argument('--segm_thresh', type=float,default=0.5)
    parser.add_argument('--test_folder')
    parser.add_argument('--videos_folder')
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
    print(net)
    print(count_parameters(net))

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

    df = {'Model':[],'Video':[],'Frame':[],'Pred_plate':[],'Confidence':[]}
    times_df = {'FOTS':[],'Classifier':[],'Refinement':[],'Detections':[]}
    with torch.no_grad():
        if args.videos_folder:
            print('Processing Videos')
            LU_table = pd.read_csv('LU_table_annotations_automatic.csv',index_col=False)
            #print(LU_table['Video_file'])
            #processed = 0
            for i,model in enumerate(sorted(os.listdir(args.videos_folder))):
                if model[0] == '.':
                    continue
                #processed+=1
                #if processed==2:break
                model_path = os.path.join(args.videos_folder,model)
                #p_videos=0
                for j,video_name in tqdm.tqdm(enumerate(sorted(os.listdir(model_path)))):
                    if video_name[0] == '.':
                        continue
                    #p_videos+=1
                    #if p_videos==2:break
                    #print(video_name)
                    video_path = os.path.join(model_path,video_name)
                    video = cv2.VideoCapture(video_path)
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames_per_second = video.get(cv2.CAP_PROP_FPS)
                    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    annotations_name = LU_table.loc[LU_table['Video_file'].apply(normalize2,form = 'NFC') == normalize('NFC',video_name)]['Annotation_file']
                    if not len(annotations_name):
                        continue
                    annotations_name = annotations_name.item()[11:-4]
                    print(f'Processing: {model}:{annotations_name}, with {num_frames} frames')
                    for k,frame in tqdm.tqdm(enumerate(frame_from_video(video))):

                        plate, confidence, times = recognize_plate(frame,net,args.segm_thresh,platenet,plateset,converter,font2,char_to_idx,device)
                        #print(plate)
                        df['Model'].append(model)
                        df['Video'].append(annotations_name)
                        df['Frame'].append(k)
                        df['Pred_plate'].append(plate)
                        df['Confidence'].append(confidence)
        elif args.test_folder:
            print('Processing Photos')
            for model in sorted(os.listdir(args.test_folder)):
                if model in ['People','Objects']:
                    continue
                lateral_path = os.path.join(args.test_folder,model,'Lateral')
                for image_name in tqdm.tqdm(sorted(os.listdir(lateral_path))):
                    path = os.path.join(lateral_path,image_name)
                    im = cv2.imread(path)
                    if args.output:
                        plate, confidence, times = recognize_plate(im,net,args.segm_thresh,platenet,plateset,converter,font2,char_to_idx,device,path,args.output)
                    else:
                        plate, confidence, times = recognize_plate(im,net,args.segm_thresh,platenet,plateset,converter,font2,char_to_idx,device)
                    for name in times:
                        times_df[name].append(times[name])
                    #print(plate, confidence)
                    df['Model'].append(model)
                    df['Video'].append(image_name[4:-9])
                    df['Frame'].append(int(image_name[-8:-4]))
                    df['Pred_plate'].append(plate)
                    df['Confidence'].append(confidence)

    pd_df = pd.DataFrame.from_dict(df,orient='columns')
    if args.videos_folder:
        pd_df.to_csv(os.path.join(args.output,'results_videos.csv'),index=False)
    elif args.test_folder:
        pd_df.to_csv(os.path.join(args.output,'results.csv'),index=False)

    pd_times_df = pd.DataFrame.from_dict(times_df,orient='columns')
    pd_times_df.to_csv('results_times.csv', index=False)
