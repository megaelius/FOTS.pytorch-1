'''
Created on 2019-06-30

@author: chenjun2hao
'''

import os
import cv2
import glob
import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from pathlib import Path
from unicodedata import normalize

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
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

def recognize_plate(im,net,device,path=None,output_folder=None):
    im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
    images = np.asarray([im_resized], dtype=float)
    images /= 128
    images -= 1
    im_data = net_utils.np_to_variable(images.transpose(0, 3, 1, 2))
    im_data = im_data.to(device)
    seg_pred, rboxs, angle_pred, features = net(im_data)

    rbox = rboxs[0].data.cpu()[0].numpy()           # 转变成h,w,c
    rbox = rbox.swapaxes(0, 1)
    rbox = rbox.swapaxes(1, 2)

    angle_pred = angle_pred[0].data.cpu()[0].numpy()

    segm = seg_pred[0].data.cpu()[0].numpy()
    segm = segm.squeeze(0)

    draw2 = np.copy(im_resized)
    boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

    img = Image.fromarray(draw2)
    draw = ImageDraw.Draw(img)

    out_boxes = []
    plate = None
    confidence = 0
    for box in boxes:

        pts  = box[0:8]
        pts = pts.reshape(4, -1)

        # det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
        det_text, conf, dec_s = align_ocr(net, converter, im_data, box, features,device, debug=0)
        if len(det_text) == 0:
          continue

        width, height = draw.textsize(det_text, font=font2)
        center =  [box[0], box[1]]
        draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
        out_boxes.append(box)
        #print(det_text, conf, dec_s)
        if is_plate(det_text) and conf > confidence:
            plate = det_text
            confidence = conf
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
    return plate, confidence

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', type=int, default=1)
    parser.add_argument('-model', default='./weights/FOTS_280000.h5')
    # parser.add_argument('-model', default='./weights/e2e-mlt.h5')
    parser.add_argument('-segm_thresh', default=0.5)
    parser.add_argument('-test_folder', default=r'./data/example_image/')
    parser.add_argument('-videos_folder', default=r'./data/example_image/')
    parser.add_argument('-output', default='./data/ICDAR2015')

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

    df = {'Model':[],'Video':[],'Frame':[],'Pred_plate':[],'Confidence':[]}
    with torch.no_grad():
        if args.videos_folder:
            LU_table = pd.read_csv('LU_table_annotations_automatic.csv',index_col=False)
            #print(LU_table['Video_file'])
            for i,model in enumerate(sorted(os.listdir(args.videos_folder))):
                if model[0] == '.':
                    continue
                if i==1:break
                model_path = os.path.join(args.videos_folder,model)
                for j,video_name in enumerate(tqdm.tqdm(sorted(os.listdir(model_path)))):
                    if video_name[0] == '.':
                        continue
                    if j==1:break
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
                        plate, confidence = recognize_plate(frame,net,device)
                        df['Model'].append(model)
                        df['Video'].append(annotations_name)
                        df['Frame'].append(k)
                        df['Pred_plate'].append(plate)
                        df['Confidence'].append(confidence)
        elif args.test_folder:
            for model in sorted(os.listdir(args.test_folder)):
                if model in ['People','Objects']:
                    continue
                lateral_path = os.path.join(args.test_folder,model,'Lateral')
                for image_name in tqdm.tqdm(sorted(os.listdir(lateral_path))):
                    path = os.path.join(lateral_path,image_name)
                    im = cv2.imread(path)

                    im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
                    images = np.asarray([im_resized], dtype=np.float)
                    images /= 128
                    images -= 1
                    im_data = net_utils.np_to_variable(images.transpose(0, 3, 1, 2), is_cuda=args.cuda)
                    seg_pred, rboxs, angle_pred, features = net(im_data)

                    rbox = rboxs[0].data.cpu()[0].numpy()           # 转变成h,w,c
                    rbox = rbox.swapaxes(0, 1)
                    rbox = rbox.swapaxes(1, 2)

                    angle_pred = angle_pred[0].data.cpu()[0].numpy()

                    segm = seg_pred[0].data.cpu()[0].numpy()
                    segm = segm.squeeze(0)

                    draw2 = np.copy(im_resized)
                    boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

                    img = Image.fromarray(draw2)
                    draw = ImageDraw.Draw(img)

                    out_boxes = []
                    plate = None
                    confidence = 0
                    for box in boxes:

                        pts  = box[0:8]
                        pts = pts.reshape(4, -1)

                        # det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
                        det_text, conf, dec_s = align_ocr(net, converter, im_data, box, features, debug=0)
                        if len(det_text) == 0:
                          continue

                        width, height = draw.textsize(det_text, font=font2)
                        center =  [box[0], box[1]]
                        draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
                        out_boxes.append(box)
                        #print(det_text, conf, dec_s)
                        if is_plate(det_text) and conf > confidence:
                            plate = det_text
                            confidence = conf
                    print(plate, confidence)
                    df['Model'].append(model)
                    df['Video'].append(image_name[4:-9])
                    df['Frame'].append(int(image_name[-8:-4]))
                    df['Pred_plate'].append(plate)
                    df['Confidence'].append(confidence)
                    im = np.array(img)
                    for box in out_boxes:
                        pts  = box[0:8]
                        pts = pts.reshape(4, -1)
                        draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

                    #cv2.imshow('img', im)
                    if not os.path.isdir(os.path.join(args.output, path.split('/')[-3])):
                        Path(os.path.join(args.output, path.split('/')[-3])).mkdir()
                    out_filename = os.path.join(args.output, path.split('/')[-3], os.path.basename(path))
                    cv2.imwrite(out_filename, im)

    pd_df = pd.DataFrame.from_dict(df,orient='columns')
    if args.videos_folder:
        pd_df.to_csv(os.path.join(args.output,'results_videos.csv'),index=False)
    elif args.test_folder:
        pd_df.to_csv(os.path.join(args.output,'results.csv'),index=False)
