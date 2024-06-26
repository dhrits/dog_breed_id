# dog_breed_id 
A Library for dog breed detection and classification

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->


## Install

``` sh
pip install https://github.com/dhrits/dog_breed_id
```

## Description
This project contains a series of models for detecting and classifying dog breeds in images. The project makes use of a combination of [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and [Tsinghua Dogs Dataset](https://cg.cs.tsinghua.edu.cn/ThuDogs/). The library itself has been built using [nbdev](https://nbdev.fast.ai/). A short guide to the repo is included below.

## How to use

### Data Preprocessing
 This project attempts to build a simple dog breed detector and classifier. It is built off of [nbdev](https://nbdev.fast.ai/), a tool used to build libraries using Jupyter notebooks. At first, please take a look at `nbs/02_data_preprocessing.ipynb` for scripts related to data preprocessing and data exploration. The notebook explores both Stanford dogs and Tsinghua datasets and combines them into a single dataset which can be utilized for training a model. It also preprocesses the two datasets and stores them in COCO format as well as a dataframe for ease of model training.

The preproessed dataset is extremely large and is thus [uploaded to S3](https://dsagar-springboard-capstone-data.s3.us-east-2.amazonaws.com/preprocessed-data/data.tar.gz). To use it, please download it in the root of the repo before executing `nbs/02_data_preprocessing.ipynb`

### Reproducing Research and Benchmarking Methods
This repository includes an attempt to reproduce research methods to do object detection and classification. Specifically, I explore [Mask R-CNN](https://arxiv.org/abs/1703.06870) a general framework for object instance segmentation. My problem of dog breed detection is simpler and only involves bounding boxes, but the approach outlined in Mask R-CNN paper is an extension of [Faster RCNN](https://arxiv.org/abs/1506.01497) which is itself the latest in the family of [RCNN (two stage object detectors)](https://medium.com/towards-data-science/exploring-object-detection-with-r-cnn-models-a-comprehensive-beginners-guide-part-2-685bc89775e2) family of detectors. This attempt is located in `nbs/03_research.ipynb`. Please download a subset of the data from [here](https://dsagar-springboard-capstone-data.s3.us-east-2.amazonaws.com/datasubset.tar.gz) to avoid recreating this data. You can download the trained model from [here](https://dsagar-springboard-capstone-data.s3.us-east-2.amazonaws.com/models/model-fasterrcnn.cuda.pt).

In addition to benchmarking FasterRCNN family of object detectors, the notebook ```04_benchmark.ipynb``` also uses [Resnet50](https://arxiv.org/abs/1512.03385) and [ConvNeXt](https://arxiv.org/abs/2201.03545) family of CNNs to test various state of the art (SOTA) CNN architectures for the purpose of dog breed detection. Based on the tradeoff between computational complexity and accuracy, Resnet50 was used for the final training.

Attribution - The code below makes use of the official [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and several libraries (for computing IOU etc) are taken from it. 
 
 ### Training on the Combined Dataset
 The notebook ```05_training.ipynb``` contains the code for training the full model. The fully trained classifier can be downloaded at [here](https://dsagar-springboard-capstone-data.s3.us-east-2.amazonaws.com/models/resnet50.pt). The code for the training is inclued in the same notebook. 

 ### Inference and Deployment
 The notebook ```06_inference.ipynb``` uses the models trained in previous notebooks to build inference code which detects, classifies and annotates dog breeds in a given image. The code developed in this notebook is then deployed in a [gradio](https://www.gradio.app/) client. This code is developed in the notebook ```07_gradio_interface.ipynb```. Finally, code for the client is developed in ```08_client.ipynb```. You may also find standalone clients in ```hf_client.py```. 

The final app is deployed at [dog_breed_id](https://huggingface.co/spaces/deman539/dog_breed_id).

### Library
Since the library was developed using nbdev, all the code from the notebooks is exported in the library included under ```dog_breed_id``` folder. The libray itself can be installed as:

``` sh
pip install https://github.com/dhrits/dog_breed_id
```
