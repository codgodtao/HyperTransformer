import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
import kornia
from kornia import laplacian, sobel
from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss


# Testing epoch.
def test(epoch):
    cc = 0.0
    sam = 0.0
    rmse = 0.0
    ergas = 0.0
    psnr = 0.0
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            image_dict, MS_image, PAN_image, reference = data

            # Generating small patches
            if config["trainer"]["is_small_patch_train"]:
                MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
                PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
                reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            # Inputs and references...
            MS_image = MS_image.float().cuda()
            PAN_image = PAN_image.float().cuda()
            reference = reference.float().cuda()

            # Taking model output
            out = model(MS_image, PAN_image)

            outputs = out["pred"]

            # Scalling
            outputs[outputs < 0] = 0.0
            outputs[outputs > 1.0] = 1.0
            outputs = torch.round(outputs * config[config["train_dataset"]]["max_value"])
            pred_dic.update({image_dict["imgs"][0] + "_pred": torch.squeeze(outputs).permute(1, 2, 0).cpu().numpy()})

            reference = torch.round(reference.detach() * config[config["train_dataset"]]["max_value"])
            pred_dic.update({image_dict["imgs"][0] + "_ref": torch.squeeze(reference).permute(1, 2, 0).cpu().numpy()})

            ### Computing performance metrics ###
            # Cross-correlation
            cc += cross_correlation(outputs, reference)
            # SAM
            sam += SAM(outputs, reference)
            # RMSE
            rmse += RMSE(outputs / torch.max(reference), reference / torch.max(reference))
            # ERGAS
            beta = torch.tensor(
                config[config["train_dataset"]]["HR_size"] / config[config["train_dataset"]]["LR_size"]).cuda()
            ergas += ERGAS(outputs, reference, beta)
            # PSNR
            psnr += PSNR(outputs, reference)

    # Taking average of performance metrics over test set
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    # Writing test results to tensorboard
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)

    # Images to tensorboard
    # Regenerating the final image
    if config["trainer"]["is_small_patch_train"]:
        outputs = outputs.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        outputs = outputs.contiguous().view(config["val_batch_size"],
                                            config[config["train_dataset"]]["spectral_bands"],
                                            config[config["train_dataset"]]["HR_size"],
                                            config[config["train_dataset"]]["HR_size"])
        reference = reference.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        reference = reference.contiguous().view(config["val_batch_size"],
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
        MS_image = MS_image.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        MS_image = MS_image.contiguous().view(config["val_batch_size"],
                                              config[config["train_dataset"]]["spectral_bands"],
                                              config[config["train_dataset"]]["HR_size"],
                                              config[config["train_dataset"]]["HR_size"])

    # Normalizing the images [0-1]
    outputs = outputs / torch.max(reference)
    reference = reference / torch.max(reference)
    MS_image = MS_image / torch.max(reference)
    if config["is_DHP_MS"] == False or config["model"] == "HyperPNN":
        MS_image = F.interpolate(MS_image, scale_factor=(
            config[config["train_dataset"]]["factor"], config[config["train_dataset"]]["factor"]), mode='bilinear')

    ms = torch.unsqueeze(MS_image.view(-1, MS_image.shape[-2], MS_image.shape[-1]), 1)
    pred = torch.unsqueeze(outputs.view(-1, outputs.shape[-2], outputs.shape[-1]), 1)
    ref = torch.unsqueeze(reference.view(-1, reference.shape[-2], reference.shape[-1]), 1)
    imgs = torch.zeros(5 * pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
    for i in range(pred.shape[0]):
        imgs[5 * i] = ms[i]
        imgs[5 * i + 1] = torch.abs(ms[i] - pred[i]) / torch.max(torch.abs(ms[i] - pred[i]))
        imgs[5 * i + 2] = pred[i]
        imgs[5 * i + 3] = ref[i]
        imgs[5 * i + 4] = torch.abs(ref[i] - ms[i]) / torch.max(torch.abs(ref[i] - ms[i]))
    imgs = torchvision.utils.make_grid(imgs, nrow=5)
    writer.add_image('Images', imgs, epoch)

    # Return Outputs
    metrics = {"cc": float(cc),
               "sam": float(sam),
               "rmse": float(rmse),
               "ergas": float(ergas),
               "psnr": float(psnr)}
    return image_dict, pred_dic, metrics


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':

    __dataset__ = {"pavia_dataset": pavia_dataset, "botswana_dataset": botswana_dataset,
                   "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset,
                   "LRHR_dataset": LRHRDataset}

    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_PANNET.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    # Loading the config file
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # Set seeds.
    torch.manual_seed(7)

    # Setting number of GPUS available for training.
    num_gpus = torch.cuda.device_count()

    # Selecting the model.
    model = MODELS[config["model"]](config)
    print(f'\n{model}\n')

    # Sending model to GPU  device.
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print("Single Cuda Node is avaiable")
        model.cuda()

    test_loader = data.DataLoader(
        __dataset__[config["train_dataset"]](
            config,
            is_train=False,
            want_DHP_MS_HR=config["is_DHP_MS"],
        ),
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=False,
    )

    # Resume...
    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    else:
        # initialize_weights(model)
        print("please use command --resume to input the path of checkpoint for inference!")
        initialize_weights_new(model)
        print("testing from scratch!!!")

    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + "/" + "N_modules(" + str(
        config["N_modules"]) + ")/inference"
    ensure_dir(PATH + "/")
    if not os.path.exists(path=PATH):
        os.makedirs(PATH)
    writer = SummaryWriter(log_dir=PATH)
    shutil.copy2(args.config, PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout
        image_dict, pred_dic, metrics = test(1)
        print(metrics)

        with open(PATH + "/" + "best_metrics.json", "w+") as outfile:
            json.dump(metrics, outfile)

        # Saving best prediction
        savemat(PATH + "/" + "final_prediction.mat", pred_dic)
