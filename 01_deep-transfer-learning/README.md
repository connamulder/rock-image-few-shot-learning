# rock-image-transfer-learning
This is the source codes for deep transfer learning of rock images.

## Preparation
- Python==3.7.3    
- Tensorflow-gpu==2.6.0
- sklearn==1.0.2

## Running the code

### 1. Deep transfer learning of rock images
The training parameters are defined in the file "system.config".
```shell
python 01_RockImageEpochTrain.py
```


### 2. Rock image dimensionality reduction visualization using t-SNE
The training parameters are defined in the system.config. The t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm in the sklearn library was used to conduct dimensionality reduction visualization of rock images.
```shell
python 02_rock_image_dataset_t-SNE.py
```

### 3. Extracting features from rock images
Parameters were setting in the function "parse_args".
```shell
python 03_save_rock_images_features.py
```
