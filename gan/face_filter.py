import face_recognition as fr
import os
import shutil
import numpy as np


thres = 1/30
img_src_path = os.path.join("..", "data_filtered")
img_dest_path = os.path.join("..", "data_filtered" + (("_thres_" + str(round(thres, 3))) if thres is not None else ""))
if not os.path.isdir(img_dest_path):
    os.mkdir(img_dest_path)

source_filenames = os.listdir(img_src_path)
keep_counter = 0
for i, img in enumerate(source_filenames):
    print("Progress: ", round(i/len(source_filenames), 3), "% retaining: ", round(keep_counter/(i+1), 3), "%")
    image = fr.load_image_file(os.path.join(img_src_path, img))
    loc = fr.face_locations(image)
    thres_px = image.shape[0] * image.shape[1] * thres if thres is not None else None
    if(len(loc) > 0):
        sizes = [(bottom - top) * (right - left) for (top, right, bottom, left) in loc]
        print(sizes)
        if thres_px is None or max(sizes) > thres_px:
            keep_counter += 1
            shutil.copy2(os.path.join(img_src_path, img), os.path.join(img_dest_path, img))

        
