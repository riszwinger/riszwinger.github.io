---
title: 'CenterNet Object Detection'
date: 2020-07-28
permalink: /posts/2020/07/CenterNet/
excerpt: CenterNet is a Deep learning based object detection model ,which detects each object as a triplet (pair of corners(top left and bottom right) and a center keypoint.
tags:
  - CenterNet
  - Object Detection
  - Deep Learning
  - Python
  - CenterNet
header:
  teaser: "/images/2020-07-28-CenterNet_files/object_detect.JPG"
---

<img src="/images/2020-07-28-CenterNet_files/object_detect.JPG">

**Object Detection** is the process of identifying/locating objects in an image or video stream and draw a bounding boxes around them.

Object Detectors are broadly classified into two categories  

1. **Two-Stage Framework ( R-CNN, Fast R-CNN, Faster R-CNN)**: In the first stage potential region of interest are extracted from the image and then in the next stage these regions are fed to the pipeline as a classification problem for object detection. 

2. **Single Stage Frameworks (YOLO, SSD)**: In the unified approach, it removes the RoI extraction process and directly classify and regress the candidate anchor boxes.

Basically, the model splits the image in the (NxN) grid and predicts bounding boxes and probabilities for each grid cell. If the midpoint of the object lies within the grid cell then the model would assign the grid cell to that object. These are typically faster than two-stage object detectors and can be used for real-time object detection.

Once we split the image in (NxN) grid, there is a possibility that the object may span over multiple grid cells and it may result in multiple detections of the same object,**non max suppression** helps to detect each object only once.

If multiple objects midpoint lies within the same grid cell then the object detector would only detect one object and would fail to detect other objects because all objects would be assigned to the same grid cell, **anchor boxes** helps to eliminate this problem by assigning each object to a grid cell and to a specific anchor box.

[CornerNet](https://arxiv.org/pdf/1808.01244.pdf) is an object detection algorithm that eliminates the use of anchor boxes and it detects an object bounding box as a pair of key points, the top-left corner, and the bottom-right corner).

### CenterNet

[CenterNet](https://arxiv.org/pdf/1904.08189.pdf) is an improved version of CornetNet, which detects each object as a triplet (pair of corners same as CornerNet (top left and bottom right) and an additional center keypoint. CenterNet uses an hourglass network as the backbone, followed by cascade corner pooling and center pooling to output two corner heatmaps (top-left & bottom-right corners) and a center keypoint heatmap, respectively. Similar to CornerNet, a pair of detected corners and
the similar embeddings are used to detect a potential bounding box. Then the detected center keypoints are used to determine the final bounding boxes.

<img src="/images/2020-07-28-CenterNet_files/arch.JPG">

- **Backbone** (Hourglass Network): The image is passed through multiple convolution layers by downsampling and upsampling to learn the global semantic representation of data.
- **Heatmap**: The heatmaps represent the locations of key points of different categories and assign a confidence score for each keypoint.
- **Embedding**: The embeddings are used to identify if two corners are from the same object. Generally, a high inner product between the top left corner embedding and bottom right corner embedding, if the two corners are from the same object.
- **Offsets**: In the hourglass network we downsampled the image which rounded some pixels, to help with the loss of information offsets are used.  

CenterNet has introduced two new modules named **cascade corner pooling** and **center pooling**, which play the roles of enriching
information collected by both top-left and bottom-right corners and providing more recognizable information at the
central regions.

**Center Pooling**: Center Pooling is used to detect center keypoints, the backbone outputs a feature map, and to determine if a pixel in the feature map is a center keypoint, it finds the maximum value in its both horizontal and vertical directions and adds them together.


<img src="/images/2020-07-28-CenterNet_files/center_img.JPG">

**Cascade Corner pooling** Corner pooling aims to find the maximum values on the boundary directions so as to determine corners but it makes corners sensitive to the edges. To eliminate this problem Cascade Corner pooling looks along a boundary to find a boundary maximum value, then looks inside along the location of the boundary maximum value to find an internal maximum value, and finally, add the two maximum values together.

<img src="/images/2020-07-28-CenterNet_files/cascade_img.JPG">


### CenterNet Model in Action ( Running in Colab )

We will use a pretrained CenterNet model from tensorflow to detect objects in a image.

We will start with setup process of downloading model and then we running it on an image.



```python

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import tensorflow_hub as hub
import requests
from object_detection.utils import label_map_util


tf.get_logger().setLevel('ERROR')
```


```python
# Clone the tensorflow models repository
!git clone --depth 1 https://github.com/tensorflow/models
```

    Cloning into 'models'...
    remote: Enumerating objects: 2185, done.[K
    remote: Counting objects: 100% (2185/2185), done.[K
    remote: Compressing objects: 100% (1887/1887), done.[K
    remote: Total 2185 (delta 525), reused 944 (delta 273), pack-reused 0[K
    Receiving objects: 100% (2185/2185), 30.41 MiB | 13.74 MiB/s, done.
    Resolving deltas: 100% (525/525), done.
    


```bash
%%bash
sudo apt install -y protobuf-compiler
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

```

    Reading package lists...
    Building dependency tree...
    Reading state information...
    protobuf-compiler is already the newest version (3.0.0-9.1ubuntu1).
    0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
    Processing /content/models/research
    Collecting avro-python3
      Downloading https://files.pythonhosted.org/packages/b2/5a/819537be46d65a01f8b8c6046ed05603fb9ef88c663b8cca840263788d58/avro-python3-1.10.0.tar.gz
    Collecting apache-beam
      Downloading https://files.pythonhosted.org/packages/ce/0e/60ce0d855df4f6b49da552dd4e5a22e10ec4766d719ef28c6c40e2ca88ba/apache_beam-2.24.0-cp36-cp36m-manylinux2010_x86_64.whl (8.6MB)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (7.0.0)
    Requirement already satisfied: lxml in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (4.2.6)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (3.2.2)
    Requirement already satisfied: Cython in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (0.29.21)
    Requirement already satisfied: contextlib2 in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (0.5.5)
    Collecting tf-slim
      Downloading https://files.pythonhosted.org/packages/02/97/b0f4a64df018ca018cc035d44f2ef08f91e2e8aa67271f6f19633a015ff7/tf_slim-1.1.0-py2.py3-none-any.whl (352kB)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.15.0)
    Requirement already satisfied: pycocotools in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (2.0.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.4.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.0.5)
    Collecting tf-models-official
      Downloading https://files.pythonhosted.org/packages/5b/33/91e5e90e3e96292717245d3fe87eb3b35b07c8a2113f2da7f482040facdb/tf_models_official-2.3.0-py2.py3-none-any.whl (840kB)
    Collecting requests<3.0.0,>=2.24.0
      Downloading https://files.pythonhosted.org/packages/45/1e/0c169c6a5381e241ba7404532c16a21d86ab872c9bed8bdcd4c423954103/requests-2.24.0-py2.py3-none-any.whl (61kB)
    Requirement already satisfied: python-dateutil<3,>=2.8.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (2.8.1)
    Collecting mock<3.0.0,>=1.0.1
      Downloading https://files.pythonhosted.org/packages/e6/35/f187bdf23be87092bd0f1200d43d23076cee4d0dec109f195173fd3ebc79/mock-2.0.0-py2.py3-none-any.whl (56kB)
    Requirement already satisfied: httplib2<0.18.0,>=0.8 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (0.17.4)
    Requirement already satisfied: pymongo<4.0.0,>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.11.0)
    Requirement already satisfied: pytz>=2018.3 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (2018.9)
    Requirement already satisfied: typing-extensions<3.8.0,>=3.7.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.7.4.3)
    Collecting hdfs<3.0.0,>=2.1.0
      Downloading https://files.pythonhosted.org/packages/82/39/2c0879b1bcfd1f6ad078eb210d09dbce21072386a3997074ee91e60ddc5a/hdfs-2.5.8.tar.gz (41kB)
    Collecting future<1.0.0,>=0.18.2
      Downloading https://files.pythonhosted.org/packages/45/0b/38b06fd9b92dc2b68d58b75f900e97884c45bedd2ff83203d933cf5851c9/future-0.18.2.tar.gz (829kB)
    Collecting fastavro<0.24,>=0.21.4
      Downloading https://files.pythonhosted.org/packages/98/8e/1d62398df5569a805d956bd96df1b2c06f973e8d3f1f7489adf9c58b2824/fastavro-0.23.6-cp36-cp36m-manylinux2010_x86_64.whl (1.4MB)
    Collecting dill<0.3.2,>=0.3.1.1
      Downloading https://files.pythonhosted.org/packages/c7/11/345f3173809cea7f1a193bfbf02403fff250a3360e0e118a1630985e547d/dill-0.3.1.1.tar.gz (151kB)
    Collecting oauth2client<4,>=2.0.1
      Downloading https://files.pythonhosted.org/packages/c0/7b/bc893e35d6ca46a72faa4b9eaac25c687ce60e1fbe978993fe2de1b0ff0d/oauth2client-3.0.0.tar.gz (77kB)
    Requirement already satisfied: protobuf<4,>=3.12.2 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.12.4)
    Requirement already satisfied: pydot<2,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.3.0)
    Requirement already satisfied: crcmod<2.0,>=1.7 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.7)
    Collecting pyarrow<0.18.0,>=0.15.1; python_version >= "3.0" or platform_system != "Windows"
      Downloading https://files.pythonhosted.org/packages/ba/3f/6cac1714fff444664603f92cb9fbe91c7ae25375880158b9e9691c4584c8/pyarrow-0.17.1-cp36-cp36m-manylinux2014_x86_64.whl (63.8MB)
    Requirement already satisfied: grpcio<2,>=1.29.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.32.0)
    Requirement already satisfied: numpy<2,>=1.14.3 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.18.5)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (1.2.0)
    Requirement already satisfied: absl-py>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from tf-slim->object-detection==0.1) (0.10.0)
    Requirement already satisfied: setuptools>=18.0 in /usr/local/lib/python3.6/dist-packages (from pycocotools->object-detection==0.1) (50.3.0)
    Requirement already satisfied: psutil>=5.4.3 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (5.4.8)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (3.13)
    Requirement already satisfied: gin-config in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.3.0)
    Requirement already satisfied: kaggle>=1.3.9 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.5.8)
    Requirement already satisfied: dataclasses in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.7)
    Requirement already satisfied: tensorflow>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (2.1.0)
    Collecting tensorflow-model-optimization>=0.2.1
      Downloading https://files.pythonhosted.org/packages/55/38/4fd48ea1bfcb0b6e36d949025200426fe9c3a8bfae029f0973d85518fa5a/tensorflow_model_optimization-0.5.0-py2.py3-none-any.whl (172kB)
    Requirement already satisfied: google-cloud-bigquery>=0.31.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.21.0)
    Requirement already satisfied: tensorflow-addons in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.8.3)
    Collecting opencv-python-headless
      Downloading https://files.pythonhosted.org/packages/b6/2a/496e06fd289c01dc21b11970be1261c87ce1cc22d5340c14b516160822a7/opencv_python_headless-4.4.0.42-cp36-cp36m-manylinux2014_x86_64.whl (36.6MB)
    Collecting py-cpuinfo>=3.3.0
      Downloading https://files.pythonhosted.org/packages/f6/f5/8e6e85ce2e9f6e05040cf0d4e26f43a4718bcc4bce988b433276d4b1a5c1/py-cpuinfo-7.0.0.tar.gz (95kB)
    Requirement already satisfied: google-api-python-client>=1.6.7 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.7.12)
    Collecting sentencepiece
      Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    Requirement already satisfied: tensorflow-hub>=0.6.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.9.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (1.24.3)
    Collecting pbr>=0.11
      Downloading https://files.pythonhosted.org/packages/c1/a3/d439f338aa90edd5ad9096cd56564b44882182150e92148eb14ceb7488ba/pbr-5.5.0-py2.py3-none-any.whl (106kB)
    Requirement already satisfied: docopt in /usr/local/lib/python3.6/dist-packages (from hdfs<3.0.0,>=2.1.0->apache-beam->object-detection==0.1) (0.6.2)
    Requirement already satisfied: pyasn1>=0.1.7 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (0.4.8)
    Requirement already satisfied: pyasn1-modules>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (0.2.8)
    Requirement already satisfied: rsa>=3.1.4 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (4.6)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle>=1.3.9->tf-models-official->object-detection==0.1) (4.41.1)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle>=1.3.9->tf-models-official->object-detection==0.1) (0.0.1)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle>=1.3.9->tf-models-official->object-detection==0.1) (4.0.1)
    Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.6.3)
    Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.3.3)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.3.0)
    Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.12.1)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.35.1)
    Requirement already satisfied: keras-preprocessing<1.2,>=1.1.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.1.2)
    Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.10.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.1.0)
    Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.2.0)
    Requirement already satisfied: tensorboard<3,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (0.24.0)
    Requirement already satisfied: promise in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (2.3)
    Requirement already satisfied: attrs>=18.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (20.2.0)
    Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow-model-optimization>=0.2.1->tf-models-official->object-detection==0.1) (0.1.5)
    Requirement already satisfied: google-cloud-core<2.0dev,>=1.0.3 in /usr/local/lib/python3.6/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (1.0.3)
    Requirement already satisfied: google-resumable-media!=0.4.0,<0.5.0dev,>=0.3.1 in /usr/local/lib/python3.6/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (0.4.1)
    Requirement already satisfied: typeguard in /usr/local/lib/python3.6/dist-packages (from tensorflow-addons->tf-models-official->object-detection==0.1) (2.7.1)
    Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (3.0.1)
    Requirement already satisfied: google-auth-httplib2>=0.0.3 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (0.0.4)
    Requirement already satisfied: google-auth>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (1.17.2)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle>=1.3.9->tf-models-official->object-detection==0.1) (1.3)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.2.2)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.7.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.4.1)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.0.1)
    Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-metadata->tensorflow-datasets->tf-models-official->object-detection==0.1) (1.52.0)
    Requirement already satisfied: google-api-core<2.0.0dev,>=1.14.0 in /usr/local/lib/python3.6/dist-packages (from google-cloud-core<2.0dev,>=1.0.3->google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (1.16.0)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth>=1.4.1->google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (4.1.1)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.7.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.3.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.1.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.1.0)
    Building wheels for collected packages: object-detection, avro-python3, hdfs, future, dill, oauth2client, py-cpuinfo
      Building wheel for object-detection (setup.py): started
      Building wheel for object-detection (setup.py): finished with status 'done'
      Created wheel for object-detection: filename=object_detection-0.1-cp36-none-any.whl size=1577697 sha256=1245c5964e9a57595cecf898b73634481005041e85b6dd3a43086b6e60e62d38
      Stored in directory: /tmp/pip-ephem-wheel-cache-uzhvzyue/wheels/94/49/4b/39b051683087a22ef7e80ec52152a27249d1a644ccf4e442ea
      Building wheel for avro-python3 (setup.py): started
      Building wheel for avro-python3 (setup.py): finished with status 'done'
      Created wheel for avro-python3: filename=avro_python3-1.10.0-cp36-none-any.whl size=43735 sha256=3a80f92d9af8b6a7db3c55f228b1bb11b627440402936a2fc67eea8121916786
      Stored in directory: /root/.cache/pip/wheels/3f/15/cd/fe4ec8b88c130393464703ee8111e2cddebdc40e1b59ea85e9
      Building wheel for hdfs (setup.py): started
      Building wheel for hdfs (setup.py): finished with status 'done'
      Created wheel for hdfs: filename=hdfs-2.5.8-cp36-none-any.whl size=33213 sha256=57241e995b3f963250863869eb9588e9fc8bf97cbb328b5445cd5cde86838c03
      Stored in directory: /root/.cache/pip/wheels/fe/a7/05/23e3699975fc20f8a30e00ac1e515ab8c61168e982abe4ce70
      Building wheel for future (setup.py): started
      Building wheel for future (setup.py): finished with status 'done'
      Created wheel for future: filename=future-0.18.2-cp36-none-any.whl size=491057 sha256=4c7659487f9d5080d1c9be230113e9d5b554da15d367e6e1f375bc1db72f1cf8
      Stored in directory: /root/.cache/pip/wheels/8b/99/a0/81daf51dcd359a9377b110a8a886b3895921802d2fc1b2397e
      Building wheel for dill (setup.py): started
      Building wheel for dill (setup.py): finished with status 'done'
      Created wheel for dill: filename=dill-0.3.1.1-cp36-none-any.whl size=78532 sha256=9a85dd95a47e7bf9af99d814d050de467e424456d6cccfe05e8e2fe79811e032
      Stored in directory: /root/.cache/pip/wheels/59/b1/91/f02e76c732915c4015ab4010f3015469866c1eb9b14058d8e7
      Building wheel for oauth2client (setup.py): started
      Building wheel for oauth2client (setup.py): finished with status 'done'
      Created wheel for oauth2client: filename=oauth2client-3.0.0-cp36-none-any.whl size=106382 sha256=845a77f43767f553e352c3b6ada7b5fe4ab2eb6b2a3a02eeccb8a7b85d531e4d
      Stored in directory: /root/.cache/pip/wheels/48/f7/87/b932f09c6335dbcf45d916937105a372ab14f353a9ca431d7d
      Building wheel for py-cpuinfo (setup.py): started
      Building wheel for py-cpuinfo (setup.py): finished with status 'done'
      Created wheel for py-cpuinfo: filename=py_cpuinfo-7.0.0-cp36-none-any.whl size=20071 sha256=5f74f77344715ecb182c70ee4e0d11c8e2870ceeeb66f75445e11e0257bbf3a0
      Stored in directory: /root/.cache/pip/wheels/f1/93/7b/127daf0c3a5a49feb2fecd468d508067c733fba5192f726ad1
    Successfully built object-detection avro-python3 hdfs future dill oauth2client py-cpuinfo
    Installing collected packages: avro-python3, requests, pbr, mock, hdfs, future, fastavro, dill, oauth2client, pyarrow, apache-beam, tf-slim, tensorflow-model-optimization, opencv-python-headless, py-cpuinfo, sentencepiece, tf-models-official, object-detection
      Found existing installation: requests 2.23.0
        Uninstalling requests-2.23.0:
          Successfully uninstalled requests-2.23.0
      Found existing installation: future 0.16.0
        Uninstalling future-0.16.0:
          Successfully uninstalled future-0.16.0
      Found existing installation: dill 0.3.2
        Uninstalling dill-0.3.2:
          Successfully uninstalled dill-0.3.2
      Found existing installation: oauth2client 4.1.3
        Uninstalling oauth2client-4.1.3:
          Successfully uninstalled oauth2client-4.1.3
      Found existing installation: pyarrow 0.14.1
        Uninstalling pyarrow-0.14.1:
          Successfully uninstalled pyarrow-0.14.1
    Successfully installed apache-beam-2.24.0 avro-python3-1.10.0 dill-0.3.1.1 fastavro-0.23.6 future-0.18.2 hdfs-2.5.8 mock-2.0.0 oauth2client-3.0.0 object-detection-0.1 opencv-python-headless-4.4.0.42 pbr-5.5.0 py-cpuinfo-7.0.0 pyarrow-0.17.1 requests-2.24.0 sentencepiece-0.1.91 tensorflow-model-optimization-0.5.0 tf-models-official-2.3.0 tf-slim-1.1.0
    

    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    
    ERROR: pydrive 1.3.1 has requirement oauth2client>=4.0.0, but you'll have oauth2client 3.0.0 which is incompatible.
    ERROR: multiprocess 0.70.10 has requirement dill>=0.3.2, but you'll have dill 0.3.1.1 which is incompatible.
    ERROR: google-colab 1.0.0 has requirement requests~=2.23.0, but you'll have requests 2.24.0 which is incompatible.
    ERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.
    ERROR: apache-beam 2.24.0 has requirement avro-python3!=1.9.2,<1.10.0,>=1.8.1; python_version >= "3.0", but you'll have avro-python3 1.10.0 which is incompatible.
    

Ignore above errors


```python
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
```


```python
category_index[1],category_index[90]
```




    ({'id': 1, 'name': 'person'}, {'id': 90, 'name': 'toothbrush'})




```python
print("Total Number of Classess=",max(list(category_index.keys())))
```

    Total Number of Classess= 90
    

This pretrained CenterNet model can detect 90 unique objects.


Download the model from tensorflow Hub, it will take couple of minutes to load the model.


```python
model=hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")

```


```python
type(model)
```




    tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject



We can use an image to run through the model, so to test it on your image just change the below url to point to your image.


```python
url="https://i.redd.it/zlcwykss93r31.jpg"
img=Image.open(requests.get(url,stream=True).raw)
img
```




![png](/images/2020-07-28-CenterNet_files/2020-07-28-CenterNet_18_0.png)




```python
(im_width, im_height) = img.size
print("Image Width=",im_width, "| Image Height=",im_height)
```

    Image Width= 2048 | Image Height= 1488
    


```python
type(img)
```




    PIL.JpegImagePlugin.JpegImageFile



As of now, the Image is of PIL type so we need to convert it onto array before we feed it into a model and these deep learning models expects image in batch so we need to reshape to add batch dimension as below


```python
img_arr=np.array(img.getdata()).reshape((1,im_height, im_width, 3)).astype(np.uint8)
img_arr.shape
```




    (1, 1488, 2048, 3)




```python
type(img_arr)
```




    numpy.ndarray



**Run the CenterNet model on the image**


```python
results=model(img_arr)
```


```python
results.keys()
```




    dict_keys(['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'])



**Model Output**

The output of the model is a dictionary with 4 elements 

1. **detection_classes** : list of objects that model was able to detect.
2. **num_detections** : Number of detections.
3. **detection_boxes** : These are the normalized cordinates *array([ymin, xmin, ymax, xmax])* of the bounding boxes. To get the original values we need to multiply by images width / height to obtain the original values of the cordinates.
4. **detection_scores**: Probability scores of the detected object in the image.  These scores are sorted from highest to lowest.



```python
results = {key:value.numpy() for key,value in results.items()}
print(results.keys())
```

    dict_keys(['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'])
    


```python
results['detection_boxes'][0][:2]
```




    array([[0.47037214, 0.27332467, 0.6770908 , 0.6363707 ],
           [0.5161023 , 0.6359347 , 0.8554727 , 0.7321054 ]], dtype=float32)



Top 5 Probability socres


```python
results['detection_scores'][0][:5]
```




    array([0.8923725 , 0.8808961 , 0.8497791 , 0.81970644, 0.8027185 ],
          dtype=float32)



Top 5 detected objects


```python
results['detection_classes'][0][:5]
```




    array([3., 1., 2., 1., 1.], dtype=float32)



To add a unique bounding box color for each class , creating a list of 90 colours 

[Source](https://stackoverflow.com/questions/9057497/how-to-generate-a-list-of-50-random-colours-in-python)


```python

import random
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
get_colors(5)
```




    ['#e8ae08', '#38eb8a', '#ed911b', '#07ddbf', '#aa17a6']




```python
color_list=get_colors(90)
color_list=sorted(color_list)
len(color_list)
```




    90



The logic used in the below code is to locate objects where the model is confident with at least a 50% probability score


```python
im2=img.copy()
drw=ImageDraw.Draw(im2)
colour_dict={}
#font = ImageFont.truetype("arial.ttf", 15)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 20)

cnt=0
for val in np.where(results['detection_scores'][0] >= 0.5)[0]:
  label=category_index[results['detection_classes'][0][val]]['name']
  if label not in colour_dict:
    colour_dict[label]=color_list[cnt]
    cnt+=1
  


  ymin = int(results['detection_boxes'][0][val][0]*im_height)
  xmin = int(results['detection_boxes'][0][val][1]*im_width)
  ymax = int(results['detection_boxes'][0][val][2]*im_height)
  xmax = int(results['detection_boxes'][0][val][3]*im_width)
  #print(ymin, xmin, ymax, xmax)
  
  drw.rectangle([(xmin,ymin),(xmax,ymax)],outline=colour_dict[label],width=5)
  
  cordinates_top_left="("+str(xmin)+','+str(ymin)+")"
  cordinates_bottom_right="("+str(xmax)+','+str(ymax)+")"
  drw.text((xmin+((xmax-xmin)//2),ymin+((ymax-ymin)//2)),label,fill="yellow",font=font)
  #drw.text((xmin+((xmax-xmin)//2),ymin), label, fill=(0, 0, 0, 15), font=font)
  drw.text((xmin,ymin),cordinates_top_left,fill="black")
  drw.text((xmax,ymax),cordinates_bottom_right,fill="black")

```


```python
im2
```




![png](/images/2020-07-28-CenterNet_files/2020-07-28-CenterNet_40_0.png)



Looks like the model was able to detect 4 unique objects in the above image

1. Car
2. Person
3. Bicycle
4. Traffic Light

Interestingly , the model was able to detect a car just from the bonnet.

### References

[CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/pdf/1904.08189.pdf)

[CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244.pdf)

[https://opencv.org/latest-trends-of-object-detection-from-cornernet-to-centernet-explained-part-i-cornernet/](https://opencv.org/latest-trends-of-object-detection-from-cornernet-to-centernet-explained-part-i-cornernet/)

[Yannic Kilcher: https://www.youtube.com/watch?v=CA8JPbJ75tY](https://www.youtube.com/watch?v=CA8JPbJ75tY)

[https://www.youtube.com/watch?v=LfUsGv-ESbc](https://www.youtube.com/watch?v=LfUsGv-ESbc)

[https://stackoverflow.com/questions/9057497/how-to-generate-a-list-of-50-random-colours-in-python](https://stackoverflow.com/questions/9057497/how-to-generate-a-list-of-50-random-colours-in-python)


[https://stackoverflow.com/questions/48343678/store-tensorflow-object-detection-api-image-output-with-boxes-in-csv-format](https://stackoverflow.com/questions/48343678/store-tensorflow-object-detection-api-image-output-with-boxes-in-csv-format)

[https://towardsdatascience.com/12-papers-you-should-read-to-understand-object-detection-in-the-deep-learning-era-3390d4a28891](https://towardsdatascience.com/12-papers-you-should-read-to-understand-object-detection-in-the-deep-learning-era-3390d4a28891)

[https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb#scrollTo=-y9R0Xllefec](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb#scrollTo=-y9R0Xllefec)


```python

```
