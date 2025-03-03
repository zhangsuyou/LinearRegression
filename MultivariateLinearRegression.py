import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linear_regression import LinearRegression  # 假设你已经实现了线性回归类

# 读取数据
data = pd.read_csv('./data/SizeVsPrice.csv')

# 划分训练集和测试集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 定义输入和输出参数
input_param_name_1 = 'size'
input_param_name_2 = 'number'
output_param_name = 'price'

# 提取训练集和测试集数据
x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# 训练线性回归模型
num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

# 绘制损失值下降曲线
plt.figure()
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('Cost')
plt.title('Gradient Descent')
plt.show()

# 生成预测数据
predictions_num = 10
x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

# 预测 z 值
z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制训练集散点
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c='b', marker='o', label='Training Set')

# 绘制测试集散点
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c='r', marker='^', label='Test Set')

# 绘制预测平面
x_pred_grid, y_pred_grid = np.meshgrid(x_axis, y_axis)
z_pred_grid = z_predictions.reshape(predictions_num, predictions_num)
ax.plot_surface(x_pred_grid, y_pred_grid, z_pred_grid, color='g', alpha=0.5, label='Prediction Plane')

# 设置标签
ax.set_xlabel(input_param_name_1)
ax.set_ylabel(input_param_name_2)
ax.set_zlabel(output_param_name)

# 显示图例
ax.legend()

# 保存图形
plt.savefig('3d_plot_matplotlib.png')


# 显示图形
plt.show()
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
from linear_regression import LinearRegression


data = pd.read_csv('./data/SizeVsPrice.csv')

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)         # .index表示这些数据的行

input_param_name_1 = 'size'
input_param_name_2 = 'number'
output_param_name = 'price'

x_train = train_data[[input_param_name_1,input_param_name_2]].values         # 将表格数据转换为数组数据
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1,input_param_name_2]].values
y_test = test_data[[output_param_name]].values

plot_training_trace = go.Scatter3d(
    x = x_train[:,0].flatten(),
    y = x_train[:,1].flatten(),
    z = y_train.flatten(),
    name = 'Training Set',
    mode = 'markers',
    marker = {
        'size':10,
        'opacity':1,
        'line':{
            'color':'rgb(255,255,255)',
            'width':1
        },
    }
)

plot_test_trace = go.Scatter3d(
    x = x_test[:,0].flatten(),
    y = x_test[:,1].flatten(),
    z = y_test.flatten(),
    name = 'Test Set',
    mode = 'markers',
    marker = {
        'size':10,
        'opacity':1,
        'line':{
            
            'color':'rgb(255,255,255)',
            'width':1
        },
    }
)

plot_layout = go.Layout(
    title = 'Date Sets',
    scene = {
        'xaxis':{'title':input_param_name_1},
        'yaxis':{'title':input_param_name_2},
        'zaxis':{'title':output_param_name}
    },
    margin={'l':0,'r':0,'b':0,'t':0}
)

plot_data = [plot_training_trace,plot_test_trace]
plot_figure = go.Figure(data=plot_data,layout=plot_layout)

plotly.offline.iplot(plot_figure)

num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train,y_train,polynomial_degree,sinusoid_degree)               #初始化
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)            #训练

print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)        #损失值下降可视化
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 10
x_min = x_train[:,0].min()
x_max = x_train[:,0].max()

y_min = x_train[:,1].min()
y_max = x_train[:,1].max()

x_axis = np.linspace(x_min,x_max,predictions_num)
y_axis = np.linspace(y_min,y_max,predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num,1))
y_predictions = np.zeros((predictions_num * predictions_num,1))

x_y_index = 0
for x_index,x_value in enumerate(x_axis):
    for y_index,y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1
        
z_predictions = linear_regression.predict(np.hstack((x_predictions,y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),  # x 坐标
    y=y_predictions.flatten(),  # y 坐标
    z=z_predictions.flatten(),  # z 坐标
    name='Prediction Plane',    # 图例名称
    mode='markers',             # 显示模式为散点
    marker={
        'size': 1,              # 点的大小
        'opacity': 0.8,         # 点的透明度
    },
    surfaceaxis=2,
)

plot_data = [plot_training_trace,plot_test_trace,plot_predictions_trace]
plot_figure = go.Figure(data=plot_data,layout=plot_layout)
plotly.offline.iplot(plot_figure)
'''