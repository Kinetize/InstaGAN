from StackGAN_Pytorch.code.trainer import GANTrainer
from StackGAN_Pytorch.code.miscc import config
import os
import numpy as np
from PIL import Image

cfg_file = os.path.join("insta_gan_gui", "insta_s2.yml")
config.cfg_from_file(cfg_file)
output_folder = os.path.join("insta_gan_gui", "output")
trainer = GANTrainer(output_folder)

batch_size = 150
condition = np.random.randn(config.cfg.Z_DIM)
condition_batch = condition.reshape(1, -1).repeat(batch_size, axis=0)
noise_batch = np.random.randn(batch_size, config.cfg.Z_DIM)

imgs_intermediate = trainer.sample_s1_image(condition_batch, noise_batch)
# imgs_final = trainer.sample_transfer(imgs_intermediate, condition_batch)

gen_img_dir = "generate_images"
try:
    os.mkdir(gen_img_dir)
except:
    pass
for i, im in enumerate(imgs_intermediate):
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    im.save(os.path.join(gen_img_dir, '{}.png'.format(i)))
