---
title: 'Data Science in Cloud'
date: 2020-01-07
permalink: /posts/2019/12/cloud_jupyter/
excerpt: This post explores how to set up an environment for training machine learning models in the cloud.
tags:
  - AWS
  - Cloud
header:
  teaser: "/images/2020-01-07-cloud_jupyter_files/header.png"
---

<img src="/images/2020-01-07-cloud_jupyter_files/header.png">


### Introduction

There are times when we are restricted by the computing power of the local machine to train the machine learning model. So as an alternative we can train our model in the cloud where we can get unlimited computing power.

In this post, we will use AWS (amazon web services) as the cloud platform to train the machine learning model in the cloud. We will start by
running up an EC2 instance, creating a Deep learning AMI, and finally running jupyter from local thereby running code on local but leveraging the unlimited computing power of the cloud.

#### Follow the below steps to set up an EC2 instance and connect from the local jupyter to train a machine learning model in the cloud 

- Start an EC2 instance by logging in the AWS platform and choose any Amazon machine image (AMI)

<img src="/images/2020-01-07-cloud_jupyter_files/ami.JPG">

- Select a GPU instance based on the data to be processed and click on configure instance details.

<img src="/images/2020-01-07-cloud_jupyter_files/instance.JPG">

- Click on Launch and select your existing Keypair.

<img src="/images/2020-01-07-cloud_jupyter_files/key_pair.JPG">

- EC2 instance will start running.

<img src="/images/2020-01-07-cloud_jupyter_files/start.JPG">

- To connect from Local through a Jupyter notebook, open cmd prompt to ssh GPU instance. Please make sure the keypair file(filename.pem) should exist in the same folder, otherwise, pass the keyPair file with the complete path.



```python
ssh -i <key-pair_filename>.pem ubuntu@<IP>
```

<img src="/images/2020-01-07-cloud_jupyter_files/cm1.JPG">

6. Run below command in command prompt to start Jupyter on the GPU instance.  and save the token 


```python
jupyter notebook --no-browser --port=8889
```

<img src="/images/2020-01-07-cloud_jupyter_files/cm2.JPG">

- Open a new command prompt to redirect Jupyter to the running GPU instance from the Local.


```python
ssh -i <key-pair_filename>.pem -L 8002:localhost:8889 ubuntu@<ip>
```

- Open a web browser and go to http://localhost:8002/tree/ and use the token that we saved earlier.
        
<img src="/images/2020-01-07-cloud_jupyter_files/jupyt.JPG">

- Jupyter notebook should start running, click on New to choose the environment.

<img src="/images/2020-01-07-cloud_jupyter_files/env.JPG">

- We can start using Jupyter notebook backed by  GPU instance for analysis.

<img src="/images/2020-01-07-cloud_jupyter_files/gpu.JPG">

- **Don't forget to turn off instance after use, go back to AWS, click on Actions → Instance State → Terminate.**


### Conclusion
 
In this post, we explored how to set up an environment for training machine learning models in the cloud.







```python

```
