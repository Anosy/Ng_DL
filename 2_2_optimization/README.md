# 本目录主要是学习各种优化方法对模型产生的影响
## 使用mini-batch梯度下降产生的结果：
Cost after epoch 0: 0.690736<br>
Cost after epoch 1000: 0.685273<br>
Cost after epoch 2000: 0.647072<br>
Cost after epoch 3000: 0.619525<br>
Cost after epoch 4000: 0.576584<br>
Cost after epoch 5000: 0.607243<br>
Cost after epoch 6000: 0.529403<br>
Cost after epoch 7000: 0.460768<br>
Cost after epoch 8000: 0.465586<br>
Cost after epoch 9000: 0.464518<br>
Accuracy: 0.796666666667<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/mini_batch_cost.png)<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/minibatch_class.png)<br>
## 使用momentum optimization 梯度下降产生的结果：
Cost after epoch 0: 0.690741<br>
Cost after epoch 1000: 0.685341<br>
Cost after epoch 2000: 0.647145<br>
Cost after epoch 3000: 0.619594<br>
Cost after epoch 4000: 0.576665<br>
Cost after epoch 5000: 0.607324<br>
Cost after epoch 6000: 0.529476<br>
Cost after epoch 7000: 0.460936<br>
Cost after epoch 8000: 0.465780<br>
Cost after epoch 9000: 0.464740<br>
Accuracy: 0.796666666667<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/moment_cost.png)<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/moment_class.png)<br>
## 使用adam优化方法进行梯度下降产生的结果：
Cost after epoch 0: 0.690552<br>
Cost after epoch 1000: 0.185567<br>
Cost after epoch 2000: 0.150852<br>
Cost after epoch 3000: 0.074454<br>
Cost after epoch 4000: 0.125936<br>
Cost after epoch 5000: 0.104235<br>
Cost after epoch 6000: 0.100552<br>
Cost after epoch 7000: 0.031601<br>
Cost after epoch 8000: 0.111709<br>
Cost after epoch 9000: 0.197648<br>
Accuracy: 0.94<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/adam_cost.png)<br>
![](https://github.com/Anosy/Ng_DL/blob/master/2_2_optimization/picture/adam_class.png)<br>
### 从结果可以看出，在adam优化算法中效果最佳，而且其损失函数平滑性最好！

