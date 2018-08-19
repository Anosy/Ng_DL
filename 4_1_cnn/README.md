# 本部分主要介绍的是使用python来实现CNN的前向传播，因为后向传播较为复杂就没有在这写出，这里还使用了tensorflow来搭建了个简单的卷积神经网络
## 第一部分的代码见CNN_forward.py
## 第二部分代码使用tensorflow来搭建了简单的卷积神经网络
### 具体的流程为：CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
### 得到的结果如下：
Cost after epoch 0: 1.906084<br>
Cost after epoch 5: 1.534486<br>
Cost after epoch 10: 0.949241<br>
Cost after epoch 15: 0.770959<br>
Cost after epoch 20: 0.640120<br>
Cost after epoch 25: 0.517365<br>
Cost after epoch 30: 0.446226<br>
Cost after epoch 35: 0.442709<br>
Cost after epoch 40: 0.358499<br>
Cost after epoch 45: 0.347636<br>
Cost after epoch 50: 0.327650<br>
Cost after epoch 55: 0.289915<br>
Cost after epoch 60: 0.264016<br>
Cost after epoch 65: 0.235301<br>
Cost after epoch 70: 0.220236<br>
Cost after epoch 75: 0.225698<br>
Cost after epoch 80: 0.208890<br>
Cost after epoch 85: 0.227749<br>
Cost after epoch 90: 0.206844<br>
Cost after epoch 95: 0.163530<br>
Train Accuracy: 0.912037<br>
Test Accuracy: 0.825<br>
![](https://github.com/Anosy/Ng_DL/blob/master/4_1_cnn/picture/cost.png)<br>
### 将迭代的次数增加到500次，得到的结果如下：
Cost after epoch 0: 1.905959<br>
Cost after epoch 5: 1.512580<br>
Cost after epoch 10: 0.991604<br>
Cost after epoch 15: 0.769688<br>
Cost after epoch 20: 0.637128<br>
Cost after epoch 25: 0.511770<br>
Cost after epoch 30: 0.440559<br>
Cost after epoch 35: 0.419136<br>
Cost after epoch 40: 0.358899<br>
...<br>
Cost after epoch 460: 0.001415<br>
Cost after epoch 465: 0.001235<br>
Cost after epoch 470: 0.001121<br>
Cost after epoch 475: 0.001075<br>
Cost after epoch 480: 0.001015<br>
Cost after epoch 485: 0.000952<br>
Cost after epoch 490: 0.000936<br>
Cost after epoch 495: 0.000865<br>
Train Accuracy: 1.0<br>
Test Accuracy: 0.89166665<br>
![](https://github.com/Anosy/Ng_DL/blob/master/4_1_cnn/picture/epoch500_cost.png)<br>
发现结果出现了过拟合的情况<br>