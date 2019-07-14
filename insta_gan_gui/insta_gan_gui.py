from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

import random
import cv2

vc = cv2.VideoCapture(0)


app = QApplication([])
window = QWidget()
window.setWindowTitle("InstaGAN")
layout = QHBoxLayout()
settings_layout = QVBoxLayout()
fps_sel_layout = QHBoxLayout()
fps_label = QLabel("FPS:")
fps_selection = QComboBox()

fps_rates = [1, 10, 30, 40]
default_fps = 3
[fps_selection.addItem(str(fps)) for fps in fps_rates]
fps_selection.setCurrentIndex(default_fps)

timer = QTimer()

def fps_selected(i):
    timer.setInterval(1000/fps_rates[i])

fps_selection.currentIndexChanged.connect(fps_selected)


cam_label = QLabel("Loading webcam image...")
gan_label = QLabel("InstaGAN")

gan_label.setAlignment(Qt.AlignCenter)


current_image = None

def timer_tick():
    if vc.isOpened():
        ret, frame = vc.read()
        if ret:
            global current_image
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_image = rgbImage
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            image = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            image = image.scaled(640, 480, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(image)
            cam_label.setPixmap(pixmap)
        else:
            print("err %s" % str(ret))
    else:
        print("not opened")

timer.timeout.connect(timer_tick)

timer.start(1000/fps_rates[default_fps])

def button_click():
    if current_image is not None:
        h, w, ch = current_image.shape
        bytesPerLine = ch * w
        image = QImage(current_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        image = image.scaled(640, 480, Qt.KeepAspectRatio)
        pixmap = QPixmap.fromImage(image)
        gan_label.setPixmap(pixmap)




button = QPushButton("Transform")
button.clicked.connect(button_click)
fps_sel_layout.addWidget(fps_label)
fps_sel_layout.addWidget(fps_selection)
settings_layout.addLayout(fps_sel_layout)
settings_layout.addWidget(button)
settings_layout.setAlignment(Qt.AlignBottom)

layout.addWidget(cam_label)
layout.addLayout(settings_layout)
layout.addWidget(gan_label)

window.setLayout(layout)

window.show()

app.exec_()