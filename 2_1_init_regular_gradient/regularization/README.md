# 本程序主要介绍的正则化对模型的影响
## non-regularization 情况下模型的结果：
Cost after iteration 0: 0.6557412523481002<br>
Cost after iteration 10000: 0.1632998752572419<br>
Cost after iteration 20000: 0.13851642423239133<br>
On the train set:<br>
Accuracy: 0.9478672985781991<br>
On the test set:<br>
Accuracy: 0.915<br>
模型的损失函数情况：<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/non-regularization.png)<br>
模型的分类结果情况: <br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/non-regularzation-classification.png)<br>
## regularization 情况下模型的结果
Cost after iteration 0: 0.6974484493131264<br>
Cost after iteration 10000: 0.2684918873282239<br>
Cost after iteration 20000: 0.2680916337127301<br>
On the train set:<br>
Accuracy: 0.9383886255924171<br>
On the test set:<br>
Accuracy: 0.93<br>
模型的损失函数情况：<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/regularization.png)<br>
模型的分类结果情况: <br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/regularzation-classification.png)<br>
## dropout 正则化模型的结果
Cost after iteration 0: 0.6543912405149825<br>
Cost after iteration 10000: 0.061016986574905605<br>
Cost after iteration 20000: 0.060582435798513114<br>
On the train set:<br>
Accuracy: 0.9289099526066351<br>
On the test set:<br>
Accuracy: 0.95<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/dropout.png)<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_1_init_regular_gradient/regularization/picture/dropout-classification.png)<br>
注意事项:dropout正则化的过程中，在训练的过程中，前向传播和后向传播都需要考虑到dropout的过程。但是在测试(预测)过程中，则不需要对dropout来去掉一些神经元！<br>