import numpy as np
from utils import prepare_for_training

"""
使用梯度下降的根本原因不是因为利用原数据算不出最优参数，至少在这里不是
但对于时刻变化的数据或者需要模拟比较复杂的模型的时候，可能就计算出来就很麻烦或者无法算出来了
而梯度下降将这一过程给简单化、可行化了
这就是它的意义
"""

class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):            #labels为真实值
        (data_processed,features_mean,features_deviation) = prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True)        #对数据进行预处理
        
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree          
        self.normalize_data = normalize_data
        
        
        num_feature = self.data.shape[1]            #检测data的特征数量（shape返回的是一个n*m的值，n表示有几行有多少组数据，m表示有几列，有几个特征值）
        self.theta = np.zeros((num_feature,1))      #通过特征值的数量，预设一个参数的空矩阵，后续对其进行更新
        
    def train(self,alpha,num_iterations = 500):      #学习率及迭代次数，alpha为学习率（步长），num_iterations为迭代次数
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations) 
        return self.theta,cost_history

    def gradient_descent(self,alpha,num_iterations):        #梯度下降（传入学习率和迭代次数）
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = []

        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))          #计算每一步的损失值并将其放入cost_history中
        return cost_history
            
    def gradient_step(self,alpha):          #每一次迭代进行的theta更新
        """
        梯度下降及参数更新，矩形运算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T   #一共num_exanples个，只乘以alpha ·3
        self.theta = theta     
        
    def cost_function(self,data,labels):
        """
        损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/num_examples     #自己定义
        return cost[0][0]
    
    @staticmethod
    def hypothesis(data,theta):
        predictions = np.dot(data,theta)        #用现在的参数theta乘上原来的data，得到预测值
        return predictions
    
    
##### 其他的一些工具
    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]      #加个[0]因为只需要data
        
        return self.cost_function(data_processed,labels)
    
    def predict(self,data):
        data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        
        return predictions