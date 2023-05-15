# rock-image-few-shot-learning
This is the source codes for few-shot learning of rock images.

## Preparation
- Python==3.7.3
- Pytorch-gpu==1.9.1
- py2neo==2021.2.3

## Running the codes and scripts

### 1. Prepare the feature split json file
Randomly select a specified number of rock images per class and save them as a json format file.
```shell
python 01_split_rock_images_to_json.py
```

### 2. Prepare json file for few-shot learning
Save few-shot learning data in a json format file.
```shell
python 02_label_idx_json_file_save.py
```

### 3. Read the rock type similarity
Read the rock type similarity knowledge from the PGKG and save it as a npy format file.
```shell
python 03_ReadSimilarityfromKG.py
```

### 4. Carry out the comparative few-shot learning experiments

#### For VGG16
```shell
 ./res_vgg16.sh
```

#### For InceptionV3
```shell
./res_inceptionv3.sh
```
