import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import scipy.misc as smi

import cv2
import imageio
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def convert_rgb_to_ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

scales = [3]

# image_list = glob.glob(opt.dataset+"_mat/*.*") 
image_list = glob.glob('../data/' + opt.dataset + '/*.bmp')

for scale in scales:
    for image_name in image_list:
        print("Processing ", image_name)

        im = imageio.imread(image_name).astype(np.float)
        # height, width, channels = im.shape
        # im = cv2.resize(im, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        # im = misc.imresize(im, size=3, interp='bicubic', mode=None)
        im_ycbcr = convert_rgb_to_ycbcr(im)
        im_y = im_ycbcr[:,:,0].astype(float)
        im_input = im_y/255.
        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])


        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        HR = model(im_input)
        HR = HR.cpu()

        im_h_y = HR.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0,:,:]

        im_h = colorize(im_h_y, im_ycbcr)

        iname = image_name.split('/')[-1].split('.')[0]
        im_h.save('outputs/' + iname + '.png')
