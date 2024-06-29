# data-mining

## 环境与路径配置

​		python 在原环境中使用的是3.12.4，在根目录下运行pip install -r requirements.txt完成基本环境配置。

​		将train_public.csv、train_internet.csv（注：需要手动将其中的“is_default”改为“isDefault”）、test_public.csv置于data目录下，整体的路径配置应该如下：

.
├── README.md
├── create_validation.py
├── data
│   ├── encoder.py
│   ├── pre_process.py
│   ├── test_public.csv
│   ├── train_internet.csv
│   ├── train_public.csv
│   ├── visualization
├── internet_select.py
├── main.py
├── models
│   ├── lightgbm.py
│   ├── logistic_regression.py
│   ├── model.py
│   ├── neural_network.py
│   ├── neural_networks
│   │   └── mlp.py
│   ├── random_forest.py
│   └── xgboost.py
├── requirements.txt
├── results
├── roc_auc.png
├── submission.csv
├── temp
└── 数据挖掘 个贷违约预测实验报告.md

## 框架介绍

​		主要分为数据预处理（data目录）和分类模型（models目录）两个部分。主函数是main.py，运行产生的测试集结果在result目录。create_validation.py主要是完成训练集和验证集的分割，internet_select.csv则是对internet数据集进行筛选用于训练。

### 数据预处理

​		数据预处理部分主要是pre_process.py，其中的DatasetBasic用于筛选internet数据集时，用public训练集训练时的数据预处理；DatasetPro则是继承DatasetBasic，修改了选用的特征，增添了在训练集中混入internet的方法，用于main.py的数据预处理。

​		visualization目录中记录了每个特征在isDefault=0/1中的分布情况

​		encoder.py中定义了若干数据预处理的encoder供pre_process.py调用，当要扩展数据预处理的部分时，只需要在encoder.py中定义新的encoder并在pre_process.py中的固定位置增加调用即可，可扩展性比较好。

### 分类模型

​		分类模型由基类model.py中定义的Model继承而来，其余具体的分类模型都是在Model的基础上实现对应的fit和predict产生的。尤其对神经网络模型，在neural_networks目录下可以自定义不同的例如MLP.py等的神经网络架构，然后在neural_network.py中固定位置替换调用。分类模型的部分我们的代码可扩展性也非常良好。

## 使用方法

### 只使用train_public.csv训练

​		首先需要分割训练集与验证集，在根目录下运行python create_validation.py即可在data目录下生成训练集train_public_split.csv，以及验证集validation_public.csv。默认是验证集占0.2，可以修改create_validation.py中的test_size参数进行调整。

​		然后可以调用不同的模型进行测试，其中对lightgbm因为进行过超参搜索，因此对其配置了roc-auc较高的参数，其他模型使用的基本上是默认参数。运行方法为在根目录下执行python main.py -M model_type，会输出在验证集上的roc-auc，并在result目录下生成测试集上的预测结果csv文件。model_type具体见如下表格：

|       模型名        | 参数名（model_type） |
| :-----------------: | :------------------: |
| logistic regression |          LR          |
|    random forest    |          RF          |
|       xgboost       |         XGB          |
|   neural network    |          NN          |
|      lightgbm       |         LGB          |

```bash
python create_validation.py
python main.py -M model_type
```

### 加上train_internet.csv训练

​		在如上分割完训练集和验证集之后，首先在public的训练集上训练并在internet上测试，internet的预测结果保存在temp目录下，筛选其中预测结果与实际标签相差最小的若干条，在data目录下生成select_train_internet.csv。使用方法为python internet_select.py -M model_type -N internet_number。其中model_type和上文中一致，internet_number参数为从internet中选取的样本数，默认为100000。internet筛选完成后，再在根目录下运行python main.py -M model_type -I True，此时会自动在训练集中加入之前筛选出的internet集合。

```bash
python create_validation.py
python internet_select.py -M model_type -N internet_number
python main.py -M model_type
```

### logistic regression手动实现

​		目前代码中默认的logistic regression是调包的版本手动实现的版本以注释的形式存在。如果要检验手动实现的版本，请将logistic_regression.py中的第一个class LogisticRegression注释掉，选择下方的第二个class LogisticRegression取消注释运行。

```bash
python create_validation.py
python main.py -M LR
```

### lightgbm超参搜索

​		目前代码中默认的lightgbm是经过超参搜索写定的较优超参，如果要进行超参搜索请将lightgbm.py中的第一个train函数注释掉，选择下方的第二个train函数取消注释运行。

```bash
python create_validation.py
python main.py -M LGB
```