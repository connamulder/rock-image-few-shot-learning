# Rock image few-shot learning
This is the source codes for rock image few-shot learning. The whole process can be divided into three stages: deep transfer learning of rock images, construction and reasoning of the petrographic knowledge graph (PGKG) and few-shot learning of rock images.
    

## 1. Stage of deep transfer learning
In the stage of deep transfer learning  of rock images, we use Tensorflow-GPU (2.6) as the development framework. The weights of VGG16, InceptionV3 and ResNet50 all were automatically downloaded by the codes of Keras applications integrated in the Tensorflow.

## 2. Stage of construction and reasoning of the PGKG
In the stage of construction and reasoning of the PGKG, the py2neo (2021.2.3) was used as the driver library and toolkit for working with the database of Neo4j (4.4.11). The plugin version of the Neo4j Graph Data Science (GDS) library installed in the database is 2.2.1. 

## 3. Stage of few-shot learning
In the stage of the few-shot learning, the Pytorch-GPU (1.9.1) was used as the development framework. 


```bash
@article{chen2023fslearning,
  title={A few-shot learning framework for rock images dually driven by data and knowledge},
  author={Chen Zhongliang,Yuan Feng, Li Xiaohui, Zhang Mingming, Zheng Chaojie},
  journal={},
  year={},
  publisher={},
  doi={}
}
```

## Contributing
Feel free to contact c_mulder@163.com if you have any question.
