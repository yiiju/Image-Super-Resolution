from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import glob

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
parser.add_argument('--gpu_ids', type=int, nargs='+', default=8,
                    help='Ids of gpus to use')
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

gpuid = ""
for i in opt.gpu_ids:
    gpuid = gpuid + str(i) + ", "
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
print("Number of device:", torch.cuda.device_count())
print("Device ids:", gpuid)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

lrlist = sorted(glob.glob(opt.image_path + '/*.png'))
for img in lrlist:
    # Prepare input
    image_tensor = Variable(transform(Image.open(img))).to(device).unsqueeze(0)

    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()

    # Save image
    fn = img.split("/")[-1]
    save_image(sr_image, f"images/outputs/{fn}")