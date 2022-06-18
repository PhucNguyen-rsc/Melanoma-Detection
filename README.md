# Melanoma Detection
## Table of contents:
1. [Overview](#1-overview)
    * [Project Summary](#11-project-summary)
    * [Relavent products mentioned](#12-relevant-products-mentioned)
    * [New changes](#13-new-changes-compared-to-our-last-time-presentation)
2. [General idea](#2-general-idea)
    * [Analysis based on TEFPA](#21-analysis-based-on-the-tefpa-model)
    * [Data](#22-data)
    * [Why we choose these CNN models?](#23-reasons-why-we-opt-out-for-those-cnn-models)
    * [Preprocess the input data](#24-preprocessing-the-input-data)
3. [Pretrained models](#3-pre-trained-model)
    * [Model 1: VGG16](#31-model-1-vgg16)
    * [Model 2: EfficientNetB0](#32-model-2-efficientnetb0)
    * [Model 3: InceptionV3](#33-model-3-inceptionv3)
    * [Model 4: ResNet50](#34-model-4-resnet50)
    * [Model 5: AlexNet (customized)](#35-model-5-alexnet-customized)
    * [Calculating weights for each model (in the ensemble phase)](#calculating-weights-for-each-model-in-the-ensemble-phase)
4. [Tools to use](#4-tools-to-use)
5. [Final result in the test set](#5-final-result-in-the-test-set-provided-by-the-competition)
6. [Deploy web application](#6-deploy-web-application-by-using-streamlit)
7. [Future directions](#7-future-directions)

## 1. Overview:
### 1.1 Project Summary
* Create a final product that can detect whether the given mole image is cancerous (melanoma disease) or not.
* "Skin cancer is a common disease that affect a big amount of peoples. Some facts about skin cancer:
    + Every year there are more new cases of skin cancer than the combined incidence of cancers of the breast, prostate, lung and colon.
    + An estimated 87,110 new cases of invasive melanoma will be diagnosed in the U.S. in 2017.
    + The estimated 5-year survival rate for patients whose melanoma is detected early is about 98 percent in the U.S. The survival rate falls to 62 percent when the disease reaches the lymph nodes, and 18 percent when the disease metastasizes to distant organs.
    + Early detection is critical!"
### 1.2 Relevant products mentioned:
  * Link trained models based on the given dataset (scroll down to see): https://drive.google.com/drive/folders/10aTvRsL1wrWjqRwsug9JHSLFOfnnqE-U?usp=sharing
  * Link of our Skype Presentation: https://drive.google.com/drive/folders/1HqhK64tP170rSeZNuFCr77HMard6Y4GJ?usp=sharing
  * Link the skin cancer Kaggle dataset that we'd used: https://www.kaggle.com/c/siim-isic-melanoma-classification/data
  
### 1.3 New changes compared to our last time presentation:
- We had added the preprocessing layer for all our models (here we only showcase the final production code of VGG16 and Alexnet, but the rest is the same). However, the result summary showcased below is still the old one (since we had not updated the new one yet)
- We also had succesfully calculated the optimal weight for each model in our dataset, as well as the final result based on the test set of the SIIM-ISIC Melanoma Classification contest on Kaggle.
- We had deployed our model using the Streamlit framework, see the video below for our final result

## 2. General idea:
* The project's aim is applying transfer learning techniques, as well as average-ensemble method to build the system that can accurately classify whether the given mole image is benign or cancerous (melanoma)
### 2.1 Analysis based on the TEFPA model:
* Task: inputs are the mole images, outputs are the final predictions (benign or malignant?)
* Experiment: mole images with their given labels
* Function space: 5 different models used as feature extractors, including AlexNet, VGG16, EfficientNetB0, InceptionV3, and ResNet50.
* Performance: used metrics are AUC, ROC; used loss function is focal crossentropy
* Algorithm: applied and retrained all deep learning pretrained models (AlexNet, VGG16, EfficientNetB0, InceptionV3, ResNet50) based on Keras/Tensorflow framework. After that, we apply the weighted average ensemble technique to intergrate all those 5 models to create a final "parent" model. We utilized the Deep Stack library to find the specific weight for each given model. 

### 2.2. Data
Sample:
   * benign mole
  
![image](https://user-images.githubusercontent.com/84164707/118297028-d487cc00-b507-11eb-903b-f185bf93d29d.png)

   * malignant mole
  
![image](https://user-images.githubusercontent.com/84164707/118296814-92f72100-b507-11eb-8578-593fed63c3ef.png)

- The dataset we had coming from the famous competition SIIM-ISIC Melanoma Classification on Kaggle (which can be accessed and downloaded via the following API: kaggle competitions download -c siim-isic-melanoma-classification).
- In principles, this dataset can be divided into 4 main parts: DICOM-formatted image dataset, JPEG-formatted image dataset, both image dataset and patients' metadata (contextual data) on the form of TFRecord, and a seperate patients' metadata and images' labels in the csv file. 
- There are two types of labels available in the dataset: 0 means benign mole images and 1 means cancerous (melanoma) mole images. The final output is a probability number that runs from 0 to 1. 
- When we were working this project, we had recently found out that there are different types of images in the benign class - sub-predictions (we initially assume that all the images belong to one single class are the same). In future, we would come up with a new strategy to deal with this problem.


![image](https://user-images.githubusercontent.com/68393604/118479274-059c1280-b73b-11eb-985a-8328ce700e95.png)

![image](https://user-images.githubusercontent.com/68393604/118479419-311efd00-b73b-11eb-8340-773ae8d072d2.png)

- The ratio between the given 2 classes (benign and melanoma) is heavily imbalanced, which is a difficult problem that must be deal with. 

![image](https://user-images.githubusercontent.com/68393604/118479771-a1c61980-b73b-11eb-8311-0baaf8936805.png)

### 2.3. Reasons why we opt out for those CNN models:
* CNN models that we used as feature extractors:
    + VGG16
    + EfficientNetB0
    + InceptionV3
    + ResNet50
    + AlexNet (we recreated this model based on the instructions in one of the Medium blogs, since Alexnet is one of the first-generation models and is not integrated into the native Keras Applications framework)
* Lý do: Based on some of the research papers about melanoma-detection that we have read, those models are regularly used in research and have been proved to give out consistent good scores)
### 2.4. Preprocessing the input data:
* Since the dataset's ratio is highly imbalanced, we had randomly selected 4000 benign mole images and 500 malignant (cancerous) mole images as our training set; we then picked randomly another 500 benign mole images + 80 malignant images as our validation set. Those datasets would be used for our training session.

![image](https://user-images.githubusercontent.com/68393604/118480020-e9e53c00-b73b-11eb-8e44-84c6548c93cd.png)

 Train dataset's ratio after we had preprocessed it

* Our team used flow_from_dataframe function alongside with Imaga Data Generator to prepare for our image data generator (for our training session).
* We used ImageDataGenerator function to resize our images into the size of 256x256, while applying various data augmentation techniques such as shearing, flipping, zooming, etc.

![image](https://user-images.githubusercontent.com/68393604/118490050-ba3c3100-b747-11eb-92de-4dd801873002.png)

## 3. Pre-trained model
* Summary: we build our deep learning models based on the following architecture: Feature extractor model (imported from Keras Applications) + customized Global Average Pooling + customized Dense layer(Prediction layer, which only has 1 node). We also applied other traning techniques to improve the models' learning process as well as their accuracy:
    + Calculating class weights (computed from sklearn) to apply to each individual class while traning {0:0.5,
                                                                                                       1:4.0)
    + Using Data Augmentation in the preparation process to help the models generalize better.
    + Using Sigmoid Focal Crossentropy as a loss function(available to use in tensorflow-addons). This loss function has been known for being extremely effective in handling imbalanced dataset, in parts because this loss function force our models to learn different patterns in the minority class.
    + Using ROC-AUC as our main metric, since this is a reliable metric used for binary classification problems that have imabalanced dataset.
    + Besides, while training, we also applied workers, use_multiprocessing to speed up the training process.
 ![image](https://user-images.githubusercontent.com/84164707/118348362-f028bc00-b573-11eb-9bf9-7c0e7b02d047.png)

### 3.1. Model 1: VGG16
* Model_name: VCG16 (feature extractor only) + customized Gloval Average Pooling 2D + customized Dense layer
* AUC score recorded in the validation set: 0.8829754330151997

![image](https://user-images.githubusercontent.com/84164707/118350526-6aac0880-b581-11eb-9c14-c56e3bc45c64.png)

### 3.2. Model 2: EfficientNetB0
* Model_name: EfficientNetB0 (feature extractor only) + customized Gloval Average Pooling 2D + customized Dense layer
* AUC score recorded in the validation set: 0.6839872746553552 

![image](https://user-images.githubusercontent.com/84164707/118350532-7a2b5180-b581-11eb-81c1-2270eaa66996.png)

### 3.3. Model 3: InceptionV3
* Model_name: InceptionV3 (feature extractor only) + customized Gloval Average Pooling 2D + customized Dense layer
* AUC score recorded in the validation set: 0.8140906680805939

![image](https://user-images.githubusercontent.com/84164707/118350544-8f07e500-b581-11eb-930b-1c1aa2b0e1d2.png)

### 3.4. Model 4: ResNet50
* Model_name: ResNet50 (feature extractor only) + customized Gloval Average Pooling 2D + customized Dense layer
* AUC score recorded in the validation set: 0.7783669141039237

![image](https://user-images.githubusercontent.com/84164707/118350581-c70f2800-b581-11eb-876c-cea9dcbc93ff.png)

### 3.5. Model 5: AlexNet (customized)
* Model_name: ResNet50 (feature extractor only) + customized Gloval Average Pooling 2D + customized Dense layer
* AUC score recorded in the validation set: 0.727487628137151

![image](https://user-images.githubusercontent.com/84164707/118350612-e1e19c80-b581-11eb-8a8c-ec35a088ef39.png)


### Calculating weights for each model (in the ensemble phase):
- With the final aim of utilizing average-ensemble technique, we had chose to use the Deep Stack library, specifically using function DirichletEnsemble() to calculate the optimal weight for each model: 0.8938 for model 1 (VGG16), 0.0403 for model 2 (EfficientNetB0), 0.0126 for model 3 (InceptionV3), 0.0029 for model 4 (ResNet50) and 0.0504 for model 5 (AlexNet)

![image](https://user-images.githubusercontent.com/68393604/118476868-3fb7e500-b738-11eb-983d-983ede716a3f.png)

--> notice: these are all old statistics before we started adding the preprocessing layer to our deep learning models (instead of relying on ImageDataGenerator function alone) and added more Dense layers after the Global Average Pooling layer.

## 4. Tools to use
* Tensorflow and Keras
* Tensorflow-addons
* Python
* Matplotlib-pyplot
* Scikit-learn
* Streamlit

![image](https://user-images.githubusercontent.com/84164707/118298422-a4412d00-b509-11eb-8abd-4f0441a00c88.png)
![image](https://user-images.githubusercontent.com/84164707/118298436-a86d4a80-b509-11eb-9b66-792f926e37bd.png)

## 5. Final result in the test set (provided by the competition):
- After submitting the predictions for the test set on Kaggle, we received a score of 0.8070, which is not an optimal score. We had suspected this low score came from the fact that we had not optimize our models as well as our training/validation set. Specifically, we needed to reselect our models (as you can see the VGG16 model accounted for the major part of the final weight), and find ways to utilize the entire given training dataset to achieve better accuracy.

## 6. Deploy web application by using Streamlit:
- We decided to use the Streamlit framework to create a local-hosted, basic website that can load and predict one image at a time (details in the Deploy file). We only use one model (instead of 5) for deployment. This will help us ensure the speed of our website while still maintaining the accuracy of the final model 

--> see our demo video of our website on Youtube: https://youtu.be/vAQ9V3F0VTA

## 7. Future directions:
* Train trên toàn bộ dữ liệu ảnh, có thể sẽ chuyển qua file TFRecord để tăng hiệu quả training lên
* Như đã nói ở trên, sẽ áp dụng lớp Preproccessing Input, thêm các lớp Dense layers phía sau
* Áp dụng các phương pháp Image Segmentation + Object Detection để nâng cao khả năng xác định được các nốt ruồi (moles) trong ảnh
* Phân tích các thang đo ROC-AUC curve và confusion matrix để xác định được threshold nên dùng trong việc xác định mole này là bening hay malignant
* Xây dựng web bắt mắt và hiệu quả hơn, dễ sử dụng cũng như tiếp cận nhiều đối tượng khác nhau.
* Nghiên cứu thực hiện nhận biết nhiều nôt ruồi cùng lúc (hợp lí trong tình huống deploy model thực tế hơn)



