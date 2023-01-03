### 实验环境

```
pip install -r requirements.txt
```

### 数据集下载

猫狗分类数据集[Download Kaggle Cats and Dogs Dataset from Official Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

### 运行方式

```
开始训练：python main.py

启动可视化工具查看训练过程：visualdl --logdir ./runs/log
```

### 实验结果

MobileNet的结果：

<img src="res\MobileNet.png" alt="avatar" style="zoom:50%;" />

融合模型的结果：

<img src="res\融合模型.png" alt="avatar" style="zoom:50%;" />

|   模型    |  ACC  |
| :-------: | :---: |
| MobileNet | 0.715 |
|  CNN+ViT  | 0.728 |
