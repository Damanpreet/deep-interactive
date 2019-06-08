# Interactive-Object-Selection TensorFlow

This is an Interative Object Selection model which utilizes Deeplab-ResNet model (Deeplab v2) as 
the backbone network. The model is trained on PASCAL VOC dataset which is publicly available online.

The original implementation of the Deeplab-Resnet used in this repository is from [here].
(https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/crf)


## Model Description

The DeepLab-ResNet is built on a fully convolutional variant of [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) with [atrous (dilated) convolutions](https://github.com/fyu/dilation), atrous spatial pyramid pooling, and multi-scale inputs (not implemented here).

The model is trained on a mini-batch of images and corresponding ground truth masks with the softmax classifier at the top. During training, the masks are downsampled to match the size of the output from the network; during inference, to acquire the output of the same size as the input, bilinear upsampling is applied. The final segmentation mask is computed using argmax over the logits.
Optionally, a fully-connected probabilistic graphical model, namely, CRF, can be applied to refine the final predictions.
On the test set of PASCAL VOC, the model achieves <code>79.7%</code> of mean intersection-over-union.

For more details on the underlying model please refer to the following paper:

    @article{CP2016Deeplab,
      title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      journal={arXiv:1606.00915},
      year={2016}
    }

The criterion to sample the positive and negative clicks for training and testing is taken from the paper.
Authors: N. Xu, B. Price, S. Cohen, J. Yang, and T. S. Huang

Deep Interactive Object Selection - N Xu et al., 2016 CVPR
- positive and negative clicks
- FCN + refinement by Graph Cut

Bibtex
If you find this code useful, please site the above two papers.


Contents:
1. Environment Setup
2. Training Model
3. Testing Model performance
4. Demo

1. Environment Setup:

The code has been tested on Ubuntu and uses Python 3.6, Tensorflow.

- Clone this repository 
https://github.com/jia2lin3yuan1/deep-interactive.git

- Setup Python environment

conda create -n objselect python==3.6 
source activate objselect

To install the required python packages, run:
pip install Tensorflow (check the correct way to install tf)
```bash
pip install -r requirements.txt
```
or for a local installation
```bash
pip install -user -r requirements.txt
```


## Caffe to TensorFlow conversion

To imitate the structure of the model, we have used `.caffemodel` files provided by the [authors](http://liangchiehchen.com/projects/DeepLabv2_resnet.html). The conversion has been performed using [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) with an additional configuration for atrous convolution and batch normalisation (since the batch normalisation provided by Caffe-tensorflow only supports inference). 
There is no need to perform the conversion yourself as you can download the already converted models - `deeplab_resnet.ckpt` (pre-trained) and `deeplab_resnet_init.ckpt` (the last layers are randomly initialised) - [here](https://drive.google.com/open?id=0B_rootXHuswsZ0E4Mjh1ZU5xZVU).

Nevertheless, it is easy to perform the conversion manually, given that the appropriate `.caffemodel` file has been downloaded, and [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) dependencies have been installed. The Caffe model definition is provided in `misc/deploy.prototxt`. 
To extract weights from `.caffemodel`, run the following:
```bash
python convert.py /path/to/deploy/prototxt --caffemodel /path/to/caffemodel --data-output-path /where/to/save/numpy/weights
```
As a result of running the command above, the model weights will be stored in `/where/to/save/numpy/weights`. To convert them to the native TensorFlow format (`.ckpt`), simply execute:
```bash
python npy2ckpt.py /where/to/save/numpy/weights --save-dir=/where/to/save/ckpt/weights
```

2. Dataset and Training

To train the network, one can use the augmented PASCAL VOC 2012 dataset with <code>10582</code> images for training and <code>1449</code> images for validation.

Copy the caffee to Tensorflow conversion

To use pretrained model, download from here. (Provide link)

- Setup the environment using the steps described above.
- Download PASCAL VOC dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data).

Your directory data structure should look like -
'''bash



Give a figure to describe the directory structure



'''

To run the sample clicks, modify the configuration file.
Configuration File path for sample clicks: 

Run the command:
python sample_clicks.py

After the above setup is complete, simply run:
```bash
python train.py 
```

To see the documentation on each of the training settings run the following:

```bash
python train.py --help
```

An additional script, `fine_tune.py`, demonstrates how to train only the last layers of the network. The script `train_msc.py` with multi-scale inputs fully resembles the training setup of the original model. 


3. Testing Model performance

- Setup the environment using the steps described above.
- The following command provides the description of each of the evaluation settings:
```bash
python evaluate.py --help
```

4. Demo

- Setup the environment using the steps described above.
- Download the pretrained model from the following link:
- To perform inference over your own images, use the following command:
```bash
python inference.py /path/to/your/image /path/to/ckpt/file
```

This will save the result with the name: 

## Missing features

The post-processing step with CRF is currently being implemented. 

    
## Other implementations
* [DeepLab-LargeFOV in TensorFlow](https://github.com/DrSleep/tensorflow-deeplab-lfov)
