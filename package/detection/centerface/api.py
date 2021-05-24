import os
import os.path as osp

import torch
import numpy as np
from PIL import Image

# local imports
from .config import Config as cfg
from .mnet import get_mobile_net
from .utils import VisionKit


def load_model():
    net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
    path = osp.join(osp.dirname(__file__), cfg.restore_model)
    weights = torch.load(path, map_location=cfg.device)
    net.load_state_dict(weights)
    net.eval()
    return net

net = load_model()


def preprocess(im):
    new_im, _, _, *params = VisionKit.letterbox(im, cfg.insize)
    return new_im, params

def postprocess(bboxes, landmarks, params):
    bboxes, landmarks = VisionKit.letterbox_inverse(*params, bboxes, landmarks, skip=2)
    return bboxes, landmarks

def detect_many(ims):
    data = torch.stack([cfg.test_transforms(im) for im in ims])
    with torch.no_grad():
        out = net(data)
    return out[0]

def decode_many(out):
    hm = out['hm']
    wh = out['wh']
    off = out['off']
    lm = out['lm']
    hm = VisionKit.nms(hm, kernel=3)
    hm.squeeze_()
    off.squeeze_()
    wh.squeeze_()
    lm.squeeze_()

    print(lm.shape)
    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    all_result = []
    for (i, hm_) in enumerate(hm):
        xs, ys = np.nonzero(hm_)
        bboxes = []
        landmarks = []
        for x, y in zip(xs, ys):
            ow = off[i][0][x, y]
            oh = off[i][1][x, y]
            cx = (ow + y) * 4
            cy = (oh + x) * 4

            w = wh[i][0][x, y]
            h = wh[i][1][x, y]
            width = np.exp(w) * 4
            height = np.exp(h) * 4

            left = cx - width / 2
            top = cy - height / 2
            right = cx + width / 2
            bottom = cy + height / 2
            bboxes.append([left, top, right, bottom])

            # landmark
            lms = []
            for j in range(0, 10, 2):
                lm_x = lm[i][j][x, y]
                lm_y = lm[i][j+1][x, y]
                lm_x = lm_x * width + left
                lm_y = lm_y * height + top
                lms += [lm_x, lm_y]
            landmarks.append(lms)
        all_result.append([bboxes, landmarks])
    return all_result


def visualize(im, bboxes, landmarks):
    return VisionKit.visualize(im, bboxes, landmarks, skip=2)


def detect_pipeline(ims):
    params_list = [] 
    new_im_list = []
    for im in ims:
        new_im, params = preprocess(im)
        new_im_list.append(new_im)
        params_list.append(params)
    preds = detect_many(new_im_list)
    box_land_list = decode_many(preds)
    final_result = []
    for (i, (bboxes, landmarks)) in enumerate(box_land_list):
        bboxes, landmarks = postprocess(bboxes, landmarks, params_list[i])
        result = {'bbox': bboxes, 'landmarks': landmarks}
        final_result.append(result)
    return final_result