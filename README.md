### Inference on Video Frames

![alt text](data/llama.png)  

Inference is using a trained model to make a prediction. Let's take the example of a convolutional neural network (CNN) to classify an image, that is, to give the image a label according to the contents of the image. If it's a picture of a Monarch Butterfly, then after the image has been input to the model, it should output ""Monarch Butterfly" as the prediction. Feeding the image into the model requires that the image have the correct shape. The notebook in this repository uses the CNN architecture called "VGG16". A VGG16 model that has been pre-trained on the 1000-class image dataset ImageNet is loaded with the line:  
```
from keras.applications.vgg16 import VGG16
```  
The training images in ImageNet are of height=224 pixels, width=224 pixels, and depth=3, so the model is expecting any input images to have this same size. In addition to this, the ImageNet training images were per-channel mean subtracted. This means that a mean value over the entire image set was calculated for each of the three image channels, Red, Green, and Blue. Then for each image in the set the Red mean was subtracted from the Red channel, the Green mean was subtracted from the Green channel, and the Blue mean was subtracted from the Blue channel.

We will have to take care of the resizing ourselves, but the per-channel mean subtraction can be done for us by importing:  
```
from keras.applications.vgg16 import preprocess_input
```  
and then calling this utility after we have extracted a frame from the video.  

In the notebook we'll see how to fit our image into an image of the appropriate size before feeding it to the input layer of the model. Prepare yourselves for llamas!

By the way, if you would like to use your own videos, or others that you stole from the web like I did, then first make sure that your object-of-interest is included  in [the list of training labels for ImageNet](http://image-net.org/challenges/LSVRC/2014/browse-synsets).



__[CLICK HERE TO SEE THE NOTEBOOK](./inference.ipynb).__