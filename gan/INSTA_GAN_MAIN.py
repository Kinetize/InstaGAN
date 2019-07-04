import INSTA_GAN
import torch
import IPython

device = torch.device(1)
hyperparameters = INSTA_GAN.hyperparameters
data_set = INSTA_GAN.InstaDataset("../img_align_celeba", hyperparameters["img_shape"])

try:
    INSTA_GAN.main(data_set, hyperparameters, device)
except:
    pass


IPython.embed()
