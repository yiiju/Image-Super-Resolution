import os
import cv2
import glob
from shutil import copyfile

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

def rename(testimgpath):
    imlist = glob.glob(testimgpath + '*.png')
    for imname in imlist:
        shortname = imname.split('/')[-1].split('.')[0]
        os.rename(imname, testimgpath + shortname + '_0.png')


def testref(testimgpath):
    imlist = glob.glob(testimgpath + '*.png')
    for imname in imlist:
        shortname = imname.split('/')[-1].split('_')[0]
        copyfile(imname, testimgpath + shortname + '_1.png')


def averagetrain(trainimgpath):
    imlist = glob.glob(trainimgpath + '*.png')
    h = 0
    w = 0
    num = 0
    for imname in imlist:
        num = num + 1
        image = cv2.imread(imname)
        height, width, channels = image.shape
        h = h + height
        w = w + width
    print('Number of images: ' + str(num))
    print('Width: ' + str(w / num))
    print('Height: ' + str(h / num))


def resizetrain(trainimgpath):
    imlist = glob.glob(trainimgpath + '*.png')
    for imname in imlist:
        shortname = imname.split('/')[-1]
        image = cv2.imread(imname)
        # image = cv2.resize(image, (384, 320), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./data/train_resize_sq/' + shortname, image)


def resizetest(testimgpath, resultimgpath):
    imlist = sorted(glob.glob(testimgpath + '*.png'))
    reslist = sorted(glob.glob(resultimgpath + '*.png'))
    i = 0
    for res in reslist:
        # shortname = res.split('/')[-1].split('_')[0]
        shortname = res.split('/')[-1].split('.')[0]
        oriimg = cv2.imread(imlist[i])
        i = i + 1
        oh, ow, channels = oriimg.shape
        resimg = cv2.imread(res)
        image = cv2.resize(resimg, (ow * 3, oh * 3), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./result/srgan/' + shortname + '.png', image)


def trainlr(trainimgpath):
    imlist = glob.glob(trainimgpath + '*.png')
    for imname in imlist:
        shortname = imname.split('/')[-1]
        image = cv2.imread(imname)
        height, width, channels = image.shape
        image = cv2.resize(image, (width // 3, height // 3), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./data/training_lr_images/' + shortname, image)


def rotatetrain(trainimgpath):
    imlist = glob.glob(trainimgpath + '*.png')
    for imname in imlist:
        shortname = imname.split('/')[-1]
        image = cv2.imread(imname)
        height, width, channels = image.shape
        if height > width:
            image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite('./data/train_rotate/' + shortname, image)


def resizebmp(testimgpath, resultimgpath):
    imlist = sorted(glob.glob(testimgpath + '*.bmp'))
    i = 0
    for orim in imlist:
        # shortname = res.split('/')[-1].split('_')[0]
        shortname = orim.split('/')[-1].split('.')[0]
        oriimg = cv2.imread(imlist[i])
        i = i + 1
        oh, ow, channels = oriimg.shape
        orimg = cv2.imread(orim)
        image = cv2.resize(orimg, (ow * 3, oh * 3), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(resultimgpath + shortname + '.bmp', image)


def calmeanstd(trainimgpath):
    # Calculate mean and std of training data
    class MyDataset(Dataset):
        def __init__(self):
            self.imgspath = []
            for imname in glob.glob(trainimgpath + '/*.png'):
                # Run in all image in folder
                self.imgspath.append(imname)

            print('Total data: {}'.format(len(self.imgspath)))

        def __getitem__(self, index):
            imgpath = self.imgspath[index]
            image = Image.open(imgpath).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((412, 412), Image.BICUBIC),
                transforms.ToTensor(),
            ])
            image = transform(image)
            return image

        def __len__(self):
            return len(self.imgspath)


    dataset = MyDataset()
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))