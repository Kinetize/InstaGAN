import INSTA_GAN
import torch
import IPython

device = torch.device(0)
hyperparameters = INSTA_GAN.hyperparameters
data_set = INSTA_GAN.InstaDataset("../data_filtered_thres_20.0", hyperparameters["img_shape"])
hyperparameters["num_images"] = 8

try:
    INSTA_GAN.main(data_set, hyperparameters, device)
except:
    pass


IPython.embed()
