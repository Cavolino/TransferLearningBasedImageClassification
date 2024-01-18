# <p align="center">Transfer Learning-Based Image Classification</p>

<p align="center">
  <a href="#track">Track</a> •
  <a href="#paper">Paper</a> •
  <a href="#technologies">Technologies</a> •
  <a href="#notebook">Notebook</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#authors">Authors</a>
</p>

**Abstract**: *The aim of the project is to try to overcome the challenge of limited data availability. For this reason has been used the method of transfer learning with the goal of improving classification performance. The target dataset is EuroSAT, a popular resource for Earth observation. The models have been pretrained on miniImageNet, a dataset commonly used in few-shot learning tasks. The project begins by processing the miniImageNet dataset with two distinct architectures that have been employed, ResNet18 and a Vision Transformer. The pre-trained models are evaluated and tested on the validation and test sets of miniImageNet. Subsequently, a small amount of images from EuroSAT has been selected to fine-tune the pretrain models and test them. To ensure robustness, fine-tuning is performed multiple times, and average results are reported. The report presents a comparative analysis of different models, including ResNet18 and Vision Transformer, evaluating their performance in the context of small datasets.*

#### Datasets: 
- [miniImageNet](https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM): a widely used dataset in machine learning and image analysis. It is designed for image classification and object recognition, specifically focusing on few-shot learning tasks. The dataset is a smaller version of the larger ImageNet dataset, containing a subset of classes selected from ImageNet. In this case only the subfolder Train was used. This folder is divided into 64 subfolders each containing 600 images 84x84 pixels
- [EuroSAT(RGB)](https://github.com/phelber/EuroSAT): dataset of satellite images designed for land cover classification. The images are captured by the Sentinel-2 satellites as part of the European Space Agency’s Copernicus Earth observation program. The ”(RGB)” indicates that the images are in the Red-Green-Blue color representation, common in color images. The dataset includes 13 land cover classes such as forests, agricultural areas, meadows, and cities.

## Track: 
1. Complete the project and submit the code. For the code, you can get help from github. (20 points)

    a) Students should try their best to improve the classification performance on EuroSAT by using some strategies like data augmentation. (You can follow these steps to complete the project)

    - i. Download and read the miniImageNet (Because the dataset miniImageNet usually is used in the few-shot learning task. Hence, the train, val, and test data are from different categories. For easier, you can only focus on the train.tar, and split the data in train.tar as train, val, and test dataset, which means you can ignore both val.tar and test.tar.) & EuroSAT(RGB) datasets. (2 points)

    - ii. Pretrain a model (ResNet10, also can be ResNet18, VGG, Vision Transformer, etc.) on the training set of miniImageNet, evaluate & test it on the validation & test set. (7 points)

    - iii. Save the pretrained model. (1 point)

    - iv. Choose 100 images from EuroSAT dataset, which are from 5 different categories and each category includes 20 samples. You should randomly choose 25 images from these 100 samples as training set (The 25 images should be from the 5 different categories. Each category includes 5 images). (3 points)

    - v. Fine-tune the pretrained model with these 25 training images and test it on the rest 75 samples, show the results. Better to fine-tuning several times on different 100 EuroSAT images and get their average result. (7 points)

    b) Compare the performance of different models (ResNet18, VGG, Vision Transformer, etc.), and investigate the different optimization strategies. (This part will be used as a bonus item, 2 points)

    c) If it’s possible, you can evaluate on the other datasets such as CropDiseases, CUB, ISIC, ChestX, etc. (This part will be used as a bonus item, 2 points)

    d) Experiments: python=3.9.18, torch=2.1.1, torchvision=0.16.1

## Paper:
[Transfer Learning-Based Image Classification Paper](./transfer_learning-based_image_classification-paper.pdf): What happens when basic deep learning methods fall short, and how can we transcend these limitations? Enter transfer learning—an innovative approach that seeks to leverage knowledge gained from one task to enhance performance on another. In practice, the learning process consists of fine-tuning a pre-trained model on the target task. Transfer learning is a valuable approach in scenarios with limited sample sizes thanks to its capability to use a pre-trained model’s early layers as a feature extractor and its capability to converge in a small amount of time since the model has already some knowledge. Overfitting is avoided as the range of data already observed during the initial training phase is wide, but at the same time, the source and the target domains must be related. 


## Technologies:
<p align="left"> 
    <a href="https://www.python.org" target="_blank" rel="noreferrer"> 
        <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> 
    </a> 
    <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> 
        <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> 
    </a> 
    <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> 
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> 
    </a>
    <a href="https://numpy.org/" target="_blank" rel="noreferrer">
        <img src="https://numpy.org/images/logo.svg" alt="NumPy" width="40" height="40" />
    </a>
    <a href="https://colab.research.google.com/" target="_blank" rel="noreferrer">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/archive/d/d0/20221103151430%21Google_Colaboratory_SVG_Logo.svg/120px-Google_Colaboratory_SVG_Logo.svg.png" alt="google_colab" width="40" height="40" />
    </a>
    <a href="https://drive.google.com/" target="_blank" rel="noreferrer">
        <img src="https://fonts.gstatic.com/s/i/productlogos/drive_2020q4/v8/web-64dp/logo_drive_2020q4_color_2x_web_64dp.png" alt="google_drive" width="40" height="40" />
    </a>
</p>

## Notebook:
[Transfer Learning-Based Image Classification Notebook](./Transfer_Learningbased_Image_Classification.ipynb)<br>
#### How To Use:
- Download both the [Datasets](#datasets).
- Download the [Notebook](./Tweets_to_Emotions.ipynb).
- Upload the downloaded dataset into [Google Drive](https://drive.google.com/) and untar/unzip using the provided code blocks in the notebook replacing the correct path.
- Replace the path of those lines (in the i, iii, iv and v points of the project):

```python
### i

path = '/content/drive/MyDrive/Deep learning/Project/kk'

### iii

torch.save(model.state_dict(), '/content/drive/MyDrive/Deep learning/Project/pretrained_resnet18.pth')

### iv

EuroSAT = "/content/drive/MyDrive/Deep learning/Project/2750"

### v

def __init__(self, file_names, class_names, root_dir = "/content/drive/MyDrive/Deep learning/Project/2750", transform=None): 
model.load_state_dict(torch.load("/content/drive/MyDrive/Deep learning/Project/pretrained_resnet18_2.pth"))

```
- Run all the Notebook.

<br><br>
<hr>
<br><br>

<div style="float: right">

##### Authors:

*Dec 2023*<br>
**Davide Moricoli, Andrea Cantore, Manex Sorarrain Agirrezabala**
</div>
