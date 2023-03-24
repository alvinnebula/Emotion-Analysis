# Emotion Detective
![python](https://img.shields.io/badge/Python-3.9.0%2B-blue)
[![View on Streamlit](https://img.shields.io/badge/Streamlit-View%20on%20Streamlit%20app-ff69b4?logo=streamlit)](https://emotion-detective.streamlit.app/)


# Directory
- [Project Overview](#project-overview)
- [Data](#data)
- [Modeling](#modeling)
- [Potential Use Cases](#potential-use-cases)
- [Resources](#resources)
- [Contributors](#contributors)

# Project Overview

> This project mainly focuses on utilizing  the technology of deep learning models to detect different emotions of people in an image. Emotion detection is a way to understand people better in social settings to detect feelings like happiness, sadness, surprise at a specific moment without actually asking them. It is useful in many areas like security, investigation and healthcare. After the algorithm was built, an emotion detection website was developed to allow users to upload images, and get the appropriate emotion of the person in the image.

# Data

Data source: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

> The data was uploaded to Google Drive from Kaggle, and it contains 35,685 emotion images in total and categorized them into 7 different categories (sad, neutral, happy, angry, disgusted, surprised, fearful). All the emotion images are saved in png format and each of them has a shape of 48x48 pixels in grayscale. However, we realized that the data labels were not all sufficiently represented: all the other emotion categories have over 4,000 image data, while for the disgusted category had about 500 images available. This can potentially cause problems later in the project. Therefore, we decided to remove the disgusted category from the data, which means only six emotions (sad, neutral, happy, angry, surprised, fearful) will be used for classification. This is done to avoid a data imbalance problem. 

# Modeling
### Modeling Approach
Model: **ResNet-50** by [**FastAI.Vision**](https://fastai1.fast.ai/vision.models.html)
```bash
from fastai.vision.widgets import *
```

Hardware accelerator: Premium GPU by [**Google Colab Pro**](https://colab.research.google.com/signup)
> We chose CNN model for this vision learning task. At the very beginning of our project, we had 3 candidates: **AlexNet**, **VGG-16**, **ResNet-50**. We tested these 3 models by training them on our emotion images and comparing their performance (including: training loss, validation loss, error rate, and training time). From the results, they all performed good on our task with same difference. But **ResNet-50** took the shortest training time (**AlexNet**: 4 hours, **VGG-16**: 2.5 hours, **ResNet-50**: 1.5 hours)

### Model Training
```bash
learn = cnn_learner(dls = path_of_images, model = models.resnet50, metrics = error_rate)

#cnn_learner(data:DataBunch, base_arch:Callable, cut:Union[int, Callable]=None, pretrained:bool=True, 
#lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[Module]=None, 
#split_on:Union[Callable, Collection[ModuleList], NoneType]=None, bn_final:bool=False, 
#init='kaiming_normal_', concat_pool:bool=True, **kwargs:Any)
```
> By **FastAI**, we can easily choose convolutional neural network by `cnn_learner`, where the **FastAI** library includes several pretrained models from [**torchvision**](https://pytorch.org/vision/stable/index.html)

> In our task, we firstly draw a learning rate plot to find a local minimum as our best learning rate. Then, we tested `epochs = 4, 8, 12, 16` respectively to make our learning cost more stable and subtle. As the training results showing us, we finally chose `epochs = 8, learning rate = 4.78e-03` to save our checking points. At last, we called interpretation functions to compare what our models learned (loss, probablilty, and performance matrix)
```bash
# Learning Rate:
learn.lr_find()

# Fine Tune:
learn.fine_tune(epochs = 8, learning = 4.78e-03)

# Interpreter:
interp = Interpretation.from_learner(learn)
interp_class = ClassificationInterpretation.from_learner(learn)
```




# Potential Use Cases 

> Although the app's functionality is based on static images. In the long run, we would love to develop a model/app that can work dynamically (i.e. take motion into account to record someone's emotion). This would mean that someone's emotional state will be recorded over the course of a certain period of time when looking into a camera. Then, the emotional states will be used to know how the person felt at different points of the conversation. They can also build emotional frequency charts to display how the frequency of the person's emotional state over the course of the conversation. Some use cases of this app are interviewing, customer support and healthcare. Interviewers can utilize the website's ability to recognize interviewees' emotions and understand what their interviewees are going through during an interview. This will assist them with dealing with other interviewees in the future. Healthcare providers can also use this website's functionalities to know what a patient was feeling during a medical treatment, so they provide care for prospective patients without subjecting them to too much pain. Customer support representatives can use this website's ability to gain knowledge about how their customers feel so they can understand how to have better conversations with other customers in the future that will satisfy their customers' demands

# Resources
- FastAI
- Streamlit
- Google Colab
- PyTorch
- MTCNN
- OpenCV
- Seaborn
- NumPy
- Matplotlib
# Contributors
- Mubarak Ganiyu (mubarak.a.ganiyu@vanderbilt.edu)
- Tinglei Wu (tinglei.wu@vanderbilt.edu)
- Alvin Chen (yiwen.chen@vanderbilt.edu)
