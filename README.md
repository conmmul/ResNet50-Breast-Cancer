**ResNet50 for Breast Cancer Histology Slides**

*Description*

This project was made for histology slides containing different tumor types to be easily detected. The slides were collected from 82 patients at zoom factors of 40x, 100x, 200x, and 400x, however, I arranged the files such that different zooms
of the same tumor type were contained in the same file to make generalization of each tumor better. Some of the images were truncated to avoid different class bias and each class had around 500-600 images each for training, validation and testing.

This is a transfer learning project used a pretrained ResNet50 model. ResNet50 is a CNN containing 50 layers and stacked residual blocks. Image preprocessing was a continuous issue throughout this project, as ResNet50 was originally trained on general classes of objecs and a transfer to
these histology slides was a large transfer. 
To mitigate that problem, I used categorical classes and unfroze all of the frames to begin the training. Then, I used *RandomFlip*, *RandomRotation*, and *RandomContrast* to increase my dataset and introduce new ways for the model to see the images.
I also used regularization penalties to the layers and the kernels to penalize the weights.

After these preprocessing steps were taken, I was able to achieve >90% test accuracy after around 20 epochs of training, freezing and unfreezing frames as needed. The whole training process to get to this point took a few days.


**How to use**

Data source: https://www.kaggle.com/datasets/ambarish/breakhis/data

* The Breast Cancer Histopathological Image Classification (BreakHis) is composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients.
* The images are collected using different magnifying factors (40X, 100X, 200X, and 400X).
* To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).
* This database has been built in collaboration with the P&D Laboratory – Pathological Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br).
* Each image filename stores information about the image itself: method of procedure biopsy, tumor class, tumor type, patient identification, and magnification factor.
* For example, SOBBTA-14-4659-40-001.png is the image 1, at magnification factor 40X, of a benign tumor of type tubular adenoma, original from the slide 14-4659, which was collected by procedure SOB.

If you would like to use and train this model on your own. All of my work was done in Jupyter notebook and you can download the code as .py and run it wherever convenient. I would recommend arranging the dataset into 8 classes: 
'adenosis','fibroadenoma','phyllodes_tumor','tubular_adenona','ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma', and 'papillary_carcinoma' and truncating the amount of images in the biased classes. This should make classification during training
and testing more accurate.
