# Image Super Resolution

An image super resolution task by using ADSR, SRGAN and ESRGAN. 

## Hardware
Ubuntu 18.04 LTS

Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

1x GeForce RTX 2080 Ti

## Set Up
### Install Dependency
All requirements is detailed in requirements.txt.

    $ pip install -r requirements.txt

### Unzip dataset
Download dataset from [Image Super Resolution](https://drive.google.com/drive/folders/1eYBSCAchU6UKLapLFU2pD_qR52c0dTmw?usp=sharing).

### Using srgan
The srgan is extend from [srgan](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan).

### Using esrgan
The esrgan is extend from [esrgan](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan).

### Using vdsr
The vdsr is extend from [vdsr](https://github.com/twtygqyy/pytorch-vdsr).

Just copy the file in [my vdsr](./vdsr) and cover the original ones.

### Using msrresnet
The msrresnet is extend from [msrresnet](https://github.com/cszn/KAIR).

Just copy the file in [my msrresnet](./msrresnet) and cover the original ones.

### Coding Style
Use PEP8 guidelines.

    $ pycodestyle *.py

## Dataset
The data directory is structured as:
```
└── data 
    ├── testing_lr_images ─ 14 test images
    └── training_hr_images ─ 291 training images
```

## Train
Train in srgan. (The root is in srgan)

Need to modify the training images into square size.

    $ sh train.sh

Argument
 - `--hr_height` the height size in hr images
 - `--hr_width` the width size in hr images

Train in esrgan. (The root is in esrgan)

Need to modify the training images into square size.

    $ sh train.sh

Argument
 - `--hr_height` the height size in hr images
 - `--hr_width` the width size in hr images

Train in vdsr. (The root is in vdsr)

    $ sh train.sh   

Train in msrresnet. (The root is in msrresnet)

    $ python main_train_msrresnet_psnr.py

## Inference
Test in srgan. (The root is in srgan)

    $ sh test.sh

Test in esrgan. (The root is in esrgan)

    $ sh test.sh

Test in vdsr. (The root is in vdsr)

    $ sh test.sh

Test in msrresnet. (The root is in msrresnet)

    $ python main_test_msrresnet.py

## Citation
```
@misc{github,
  author={eriklindernoren},
  title={PyTorch-GAN},
  year={2019},
  url={https://github.com/eriklindernoren/PyTorch-GAN},
}

@misc{github,
  author={twtygqyy},
  title={pytorch-vdsr},
  year={2018},
  url={https://github.com/twtygqyy/pytorch-vdsr},
}

@misc{github,
  author={cszn},
  title={KAIR},
  year={2020},
  url={https://github.com/cszn/KAIR},
}
```