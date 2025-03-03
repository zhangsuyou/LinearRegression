import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('./data/SizeVsPrice.csv')
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)         # .index表示这些数据的行

input_param_name = 'size'
output_param_name = 'price'

x_train = train_data[[input_param_name]].values         # 将表格数据转换为数组数据
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values


plt.scatter(x_train,y_train,label = 'train data')
plt.scatter(x_test,y_test,label = 'test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('price')
plt.legend()
plt.show()

num_iterations = 1000
learning_rate = 0.01

linear_regression = LinearRegression(x_train,y_train)               #初始化
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)            #训练

print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)        #损失值下降可视化
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label = 'train data')
plt.scatter(x_test,y_test,label = 'test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('price')
plt.legend()
plt.show()