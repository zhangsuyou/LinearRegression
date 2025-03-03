import numpy as np


def prepare_for_training(data,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        num_examples = data.shape[0]

        data_processed = np.copy(data)
        
        features_mean = 0        
        features_deviation = 0
        data_normalized = data_processed
        if normalize_data:
                (
                data_normalized,
                features_mean,
                features_deviation
                ) = normalize(data_processed)
                
                data_processed = data_normalized
        
        if sinusoid_degree > 0:
                pass
        
        if polynomial_degree >0:
                pass
        
        data_processed = np.hstack((np.ones((num_examples,1)),data_processed))
        
        return data_processed,features_mean,features_deviation

def normalize(features):                
        #进行的是减均值的过程，使其以原点为中心对称
        features_normalized = np.copy(features).astype(float)

        features_mean = np.mean(features,0)     #计算均值

        features_deviation = np.std(features,0)         #计算标准差
        
        if features.shape[0] > 1:
                features_normalized -= features_mean
                
        features_deviation[features_deviation == 0] = 1
        features_normalized /= features_deviation
        
        return features_normalized,features_mean,features_deviation

        