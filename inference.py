'''
Below we use a VGG-16 model, pre-traied on a large, 1000-classes dataset. We then
see how to input frames from a video into the model for inference. If a frame
contains one of the objects from the original dataset, the model should "predict" it.
We will see that the video frame has to be reshaped and scaled to be as like the
training data as possible.

tips:
    Below we use a VGG-16 model, pre-traied on a large, 1000-classes dataset. We then
see how to input frames from a video into the model for inference. If a frame
contains one of the objects from the original dataset, the model should "predict" it.
We will see that the video frame has to be reshaped and scaled to be as like the
training data as possible.

check if your the pre-trained model was trained on your object:
    http://image-net.org/challenges/LSVRC/2014/browse-synsets
'''

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
import skvideo.io
from skimage.transform import resize
from utils import custom_plots

model = VGG16()
print(model.summary())

infile = './data/llama.mp4'
videodata = skvideo.io.vread(infile)
N_frames, rows, cols, channels = videodata.shape
# The image must be 224x224 pixels in order to input to the model, but our image is not
# square. Solution: Create a blank (all-zeros) image, resize the width of our image to 224, and the
# resize the height by the same factor used to resize the width. Then place this into the blank,
# square one. There will be black bars at the top and bottom of the square input image because our
# frame width is greater than its height, but this will not affect our classifier.
input_rows, input_cols = (224, 224)
Img = np.zeros((input_rows, input_cols, channels), np.uint8)
horz_scaling = input_cols/cols
rz_rows = int(rows * horz_scaling)
rz_cols = input_cols
nb_blank_rows = input_rows - rz_rows
blank_height = nb_blank_rows // 2

for iframe in range(N_frames):
    if iframe%10==0:
        imgnumb = str(iframe).zfill(4)
        
        # get the current frame
        Input = videodata[iframe,:,:,:] # shape = (480, 854, 3)
        Input = resize(Input, (rz_rows, rz_cols, channels))
        maxval = np.max(Input)
        Input /= maxval
        Input *= 255
        Input = Input.astype(np.uint8)
        Img[blank_height:blank_height+rz_rows, :] = Input
        custom_plots.show_img(Img, imgnumb, pause=True)
        Input = np.expand_dims(Img, axis=0)
        Input = preprocess_input(Input)
        pred = model.predict(Input)
        label = decode_predictions(pred)
        label = label[0][0]
        # print the classification
        print(f'frame: {iframe} | prediction: {label[1] }{label[2]*100:0.2f}')
        
