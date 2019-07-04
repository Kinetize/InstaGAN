import INSTA_WGAN
import torch
import IPython

device = torch.device(1)
hyperparameters = INSTA_WGAN.hyperparameters
data_set = INSTA_WGAN.InstaDataset("../data_filtered_thres_20.0", hyperparameters["img_shape"])

try:
    INSTA_WGAN.main(data_set, hyperparameters, device)
except:
    pass


IPython.embed()
