# rock-image-few-shot-learning
This is the source codes for few-shot learning of rock images and and the cypher scripts for construction and reasoning of LithoKG.
## Preparation
    Python==3.7.3
    Pytorch-gpu==1.9.1
    py2neo==2021.2.3
## Construction and reasoning of the LithoKG
Execute the cypher scripts in turn in the neo4j browser.
## Running the codes and scripts
### 1.Prepare the feature split json file
Run 01_split_features.py. 
### 2.Prepare few-shot learning json file
Run 01_split_features.py. Save few-shot learning data file in format
### 3.Read the rock type similarity
Run 01_split_features.py.
./res_vgg16.sh

./res_inceptionv3.sh
