# LAB5 讲解

### 实验目的

考察完整进行实验的能力。此处“实验”的含义是指整个任务的过程，不单单指模型的构建。

一般来说，我们整个分类任务可以分为以下部分，供你参考：

1. 获取数据集，对数据进行分析

> 数据集我们已经给出，对于数据的部分特征我们已经给出，你也可以针对你发现的其他特点进行说明和处理。

2. 对数据进行处理，形成测试集和测试集

> 针对你发现的问题，选择合适的处理方式，推荐使用搜索引擎和课程内容 Chap. 2 & 11。

3. 对于任务，选择合适的模型

> 每个模型实际上包含多种对于不同任务的处理方式，关键的是算法的核心。
>
> 例如，在决策树中，实际上存在分类树和回归树两种，你应该选择更合适的方法

4. 利用训练集训练模型，调整超参数以达到在测试集上达到更好的效果，保存模型

> 请注意各个模型有哪些超参数是可以调整的，体现你是通过对比后选出更好的模型。
>
> 注意，调整参数不应该是一个做样子的过程，需要达到“充分”调参。
>
> 如果数量太多可以以图的形式表示。
>
> 尽管我们不需要同学提交模型，但是希望大家学会、养成保存模型的良好习惯。

5. 注意你的评价指标是否合适，同时进行假设检验。

> 参考 Chap 2.

6. 可视化你的结果
7. 挑选一个最好的模型，用其对 test_label 进行预测，提交你的 pred.

> 只需要提交你认为表现最好的模型的预测结果。



1. 对于数据的处理、数据集的划分、模型的验证等内容可以使用已有的库函数，调用的函数一定是要有具体的、有针对性的功能，而不是一个“万能函数”
2. 实验结果正确命名后，与实验报告、代码压缩后提交
3. 调参可以手动调，也可以自动调参数
4. 由于 lab3 我们写的是回归树和回归树为基分类器的 xgboost，我们对于决策树和 xgbclassifier 也可以调包。神经网络也可以调包。具体reference 为：
   1. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
   2. https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
   3. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

主题: 顾言午(YanwuGu)的快速会议
日期: 2022-12-24 19:49:37
录制文件：https://meeting.tencent.com/v2/cloud-record/share?id=7fca8550-295d-446a-bf1c-0efc3e52c00a&from=3
