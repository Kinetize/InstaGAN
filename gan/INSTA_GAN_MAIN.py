import INSTA_GAN
import torch
import IPython

data_set = INSTA_GAN.load_data_set()
device = torch.device(0)
hyperparameters = INSTA_GAN.hyperparameters
try:
    INSTA_GAN.main(data_set, hyperparameters, device)
except:
    pass


IPython.embed()
