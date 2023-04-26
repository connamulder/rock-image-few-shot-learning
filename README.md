# rock-image-few-shot-learning
This is the source codes for few-shot learning of rock images and and the cypher scripts for construction and reasoning of the LithoKG.
## Preparation
    Python==3.7.3
    Pytorch-gpu==1.9.1
    py2neo==2021.2.3
## Construction and reasoning of the LithoKG
Execute the cypher scripts in turn in the Neo4j browser. 
## Running the codes and scripts
### 1. Prepare the feature split json file
Run '01_split_rock_images_to_json.py' python file. Randomly select a specified number of rock images per class and save them as a json format file.
### 2. Prepare json file for few-shot learning
Run '02_label_idx_json_file_save.py' python file. Save few-shot learning data in a json format file.
### 3. Read the rock type similarity
Run '03_ReadSimilarityfromKG.py' python file. Read the rock type similarity knowledge from the LithoKG and save it as a npy format file.
### 4. Carry out the comparative few-shot learning experiments
#### For VGG16
    ./res_vgg16.sh
#### For InceptionV3
    ./res_inceptionv3.sh
## Contributing
Feel free to contact c_mulder@163.com if you have any question.
