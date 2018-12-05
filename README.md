Author: Jiachuan Deng

Using CNN to handel with textual data combine with meta data to build a regression model for predicting commodity price.

### Data Set
Data set is from Kaggle: https://www.kaggle.com/c/mercari-price-suggestion-challenge/data

Pre-trained W2V is used: http://nlp.stanford.edu/data/glove.6B.zip

### Files Description
> DPLModel4textualRegression.pdf: detailed paper format description for our model
> config.ini: you can change your data path and model parameters in this file

> dataprocessing.py: script to process data into format that can be directly fed into model

> minibatcher.py: helper script to load data in batch

> network.py: define the model

> run_model.py: main executable code to train/test model

> runme.py: use this script to run all things together

### Model
<p align="center"> 
<img src="https://user-images.githubusercontent.com/20760190/48676214-a8a34f80-eb31-11e8-8664-48721b5ba2f2.png" width="600">
 </p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/20760190/48676213-a80ab900-eb31-11e8-889f-ae305ca5d614.png" width="600">
 </p>
Model can be built based on RNN structure or CNN structure. Here we only provide code for CNN structure, whose complexity is lower and performs better.

### Reference
> Convolutional Neural Networks for Sentence Classification: https://arxiv.org/abs/1408.5882

> Non-Linear Text Regression with a Deep Convolutional Neural Network: http://anthology.aclweb.org/P/P15/P15-2030.pdf

> Empirical Bayes method for categorical features: http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.pdf

### Run Code:
run command : python3 runme.py


