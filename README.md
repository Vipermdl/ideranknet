## IDE Ranknet

This is the repo of paper [IDE RankNet: Estimating the difficulty of visual search in an image](paper/IDE RankNet.pdf)

**Abstract**

Estimating the difficulty of visual search in images is an interesting work that can be applied to weakly supervised object localization and semi-supervised object classification, and has potential applications in object detection. In this paper, we proposed a simple loss function based on learning to rank and applied it to an end-to-end multitask neural network for estimating difficulty scores of images. Our model shows better results for predicting the ground-truth visual search difficulty scores produced by human annotators in PASCAL VOC2012.

**Requirements**

1. python 3.0+

2. pytorch 0.3

3. visdom

4. opencv2

**Installation**

  git clone https://github.com/Vipermdl/ideranknet

**Details**

1. Dataset:
The images from PASCAL VOC2012 in the benchmark dataset are used as the difficulty dimensioning source. The dataset contains a total of 11540 training and test images, and the dataset contains 20 objects (including airplanes, boats, cats, dogs, etc.) that have been labeled with categories, outlines and borders. This task is on a crowd-sourcing platform named CrowdFlower, after 736 trusted annotator observe the information in the image, the time required to answer the question is used as a measure of the image difficulty and convert it into image difficulty score. They designed a series of related processing methods, such as clearing outliers to ensure data reality. Based on the usual visual search tasks, they propose an explanation of the difficulty of the image close to the human visual level. Since the background of each image in the PASCAL VOC2012 dataset is different, the density of the objects is different, the number, size and appearance of the objects are different, we can apply this dataset to our visual search task.
In this paper, the dataset is divided into training, validation, and test sets in a 2:1:1 ratio using the same way as the previous work. We trained our model using 5,770 images in the training set, and use others to test and validate the MSE and Kendall's τ correlation coefficients.

2. train and test: python train.py

3. Metrics:
MSE and [Kendall's corelation efficients](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

4. Results:
As follow investigation of IDE RankNet with different settings on test dataset 

<div align="center"><img src="/papers/result.png" /></div>