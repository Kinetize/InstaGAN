from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QPlainTextEdit, QLineEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from StackGAN_Pytorch.code.trainer import GANTrainer
from StackGAN_Pytorch.code.miscc import config

cfg_file = os.path.join("insta_gan_gui", "insta_s2.yml")
config.cfg_from_file(cfg_file)
output_folder = os.path.join("insta_gan_gui", "output")
trainer = GANTrainer(output_folder)

output_img_size = 256*2





import random
import cv2

vc = cv2.VideoCapture(0)


app = QApplication([])
window = QWidget()
standard_window_title = "InstaGAN"
window.setWindowTitle(standard_window_title)
layout = QHBoxLayout()
settings_layout = QVBoxLayout()
fps_sel_layout = QHBoxLayout()
fps_label = QLabel("FPS:")
fps_selection = QComboBox()
hashtags_label = QLabel("Hashtags:")
hashtags_input = QPlainTextEdit()
embeddings_factor_input = QComboBox()
embeddings_factors = np.arange(0, 100, 0.5)


embeddings_factor_input_layout = QHBoxLayout()
embeddings_factor_input_label = QLabel("Embedding x")
[embeddings_factor_input.addItem(str(f)) for f in embeddings_factors]
embeddings_factor_input.setCurrentIndex(2)
embeddings_factor_input_layout.addWidget(embeddings_factor_input_label)
embeddings_factor_input_layout.addWidget(embeddings_factor_input)

fps_rates = [1, 10, 30, 40]
default_fps = 3
[fps_selection.addItem(str(fps)) for fps in fps_rates]
fps_selection.setCurrentIndex(default_fps)


batch_sizes = [2, 5, 10, 16, 32, 64, 128]
batch_size_input_layout = QHBoxLayout()
batch_size_input_label = QLabel("Batch-Size")
batch_size_input = QComboBox()
[batch_size_input.addItem(str(bs)) for bs in batch_sizes]
batch_size_input.setCurrentIndex(2)
batch_size_input_layout.addWidget(batch_size_input_label)
batch_size_input_layout.addWidget(batch_size_input)



generator_flag_input = QCheckBox("Generator")
generator_flag_input.setCheckState(False)

timer = QTimer()


embeddings = None
def get_embeddings():
    global embeddings
    if embeddings is None:
        words = []
        vectors = []
        embedding_file_path = os.path.join("insta_gan_gui", "glove.twitter.27B.100d.txt")
        window.setWindowTitle("Loading embeddings...")
        for i in range(5):
            app.processEvents()
        with open(embedding_file_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
        embeddings = {w: vectors[i] for i, w in enumerate(words)}
        window.setWindowTitle(standard_window_title)

        return embeddings
    else:
        return embeddings

def fps_selected(i):
    timer.setInterval(1000/fps_rates[i])

fps_selection.currentIndexChanged.connect(fps_selected)


cam_label = QLabel("Loading webcam image...")
gan_label = QLabel("InstaGAN")

gan_label.setAlignment(Qt.AlignCenter)


current_image = None
freeze = False

def timer_tick():
    if not (freeze or generator_flag_input.isChecked()):
        if vc.isOpened():
            ret, frame = vc.read()
            if ret:
                global current_image
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shape_min = min(rgbImage.shape[:2])
                shape_max = max(rgbImage.shape[:2])
                offset = int((shape_max - shape_min) / 2)
                rgbImage_cropped = rgbImage[:, offset:shape_min+offset].copy()
                rgbImage_resized = cv2.resize(rgbImage_cropped, (64, 64)).copy()
                current_image = rgbImage_resized
                h, w, ch = rgbImage_resized.shape
                bytesPerLine = ch * w
                image = QImage(rgbImage_resized.data, w, h, bytesPerLine, QImage.Format_RGB888)
                image_upscaled = image.scaled(output_img_size, output_img_size)
                pixmap = QPixmap.fromImage(image_upscaled)
                cam_label.setPixmap(pixmap)
            else:
                print("err %s" % str(ret))
        else:
            print("not opened")

timer.timeout.connect(timer_tick)

timer.start(1000/fps_rates[default_fps])


image_backlog = []

def clear_backlog():
    global image_backlog
    image_backlog = []

def show_next_in_backlog():
    if len(image_backlog) > 0:
        data = image_backlog.pop(0)
        if len(data) == 2:
            img_intermediate, img_final = data
            gan_label.setPixmap(np_to_q(img_final))
            cam_label.setPixmap(np_to_q(img_intermediate))

        else:
            img_final = data[0]
            gan_label.setPixmap(np_to_q(img_final))
    next_button_show()


def np_to_q(im, output_img_size=256*2):
    img_output = ((im + 1) * 127.5).astype(np.uint8)
    h, w, ch = img_output.shape
    bytesPerLine = ch * w
    image = QImage(img_output.copy().data, w, h, bytesPerLine, QImage.Format_RGB888)
    image_scaled = image.scaled(output_img_size, output_img_size)
    pixmap = QPixmap.fromImage(image_scaled)
    return pixmap

def next_button_show():
    next_button.setText("Next (%s)" % str(len(image_backlog)))
    if len(image_backlog) > 0:
        next_button.show()
    else:
        next_button.hide()

def button_click():
    global image_backlog
    clear_backlog()

    #Read parameters
    hashtag_string_raw = hashtags_input.toPlainText()
    hashtags = list(filter(lambda h: len(h) > 0, reduce(lambda a,c: a + c.split("\n"), hashtag_string_raw.split(","), [])))
    embeddings_factor = embeddings_factors[embeddings_factor_input.currentIndex()]
    print("Embeddings factor: %s" % str(embeddings_factor))

    #Generate condition
    condition = np.zeros(100)#np.random.randn(100) * embeddings_factor
    for i in range(10):
        condition[np.random.randint(0, 100)] = 1

    if len(hashtags) > 0:
        print("Hashtags: " + str(hashtags))
        embeddings_dict = get_embeddings()
        embeddings = np.array([embeddings_dict[h] for h in hashtags if h in embeddings_dict])
        condition = embeddings.mean(axis=0) * embeddings_factor
        print(condition)



    #generating batch
    batch_size = batch_sizes[batch_size_input.currentIndex()]
    condition_batch = condition.reshape(1, -1).repeat(batch_size, axis=0)
    noise_batch = np.random.randn(batch_size, config.cfg.Z_DIM)

    if generator_flag_input.isChecked():
        imgs_intermediate = trainer.sample_s1_image(condition_batch, noise_batch)
    else:
        if current_image is not None:
            imgs_intermediate = current_image/255.0 * 2.0 - 1
            imgs_intermediate = np.expand_dims(imgs_intermediate, axis=0)
            condition_batch = condition_batch[:1, :]
        else:
            imgs_intermediate = None

    if imgs_intermediate is not None:
        imgs_final = trainer.sample_transfer(imgs_intermediate, condition_batch).copy()

    if generator_flag_input.isChecked():
        image_backlog += list(zip(imgs_intermediate, imgs_final))
    else:
        image_backlog += list((img,) for img in imgs_final)
    show_next_in_backlog()





def freeze_callback():
    global freeze
    freeze = not freeze


button = QPushButton("Transform")
button.clicked.connect(button_click)

next_button = QPushButton("Next")
next_button.clicked.connect(show_next_in_backlog)
next_button.hide()

freeze_button = QPushButton("Freeze")
freeze_button.clicked.connect(freeze_callback)

button_layout = QHBoxLayout()
button_layout.addWidget(button)
button_layout.addWidget(next_button)
fps_sel_layout.addWidget(fps_label)
fps_sel_layout.addWidget(fps_selection)
settings_layout.addWidget(hashtags_label)
settings_layout.addWidget(hashtags_input)
settings_layout.addLayout(fps_sel_layout)
settings_layout.addLayout(embeddings_factor_input_layout)
settings_layout.addWidget(generator_flag_input)
settings_layout.addLayout(batch_size_input_layout)
settings_layout.addWidget(freeze_button)
settings_layout.addLayout(button_layout)
settings_layout.setAlignment(Qt.AlignBottom)

layout.addWidget(cam_label)
layout.addLayout(settings_layout)
layout.addWidget(gan_label)

window.setLayout(layout)

window.show()

app.exec_()