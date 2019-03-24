from evaluation import evaluate_nn
from data_retrieval import get_data
from os import environ
from tensorflow import set_random_seed
from numpy.random import seed

np_seed = 61619
tf_seed = 25383
seed(np_seed)
set_random_seed(tf_seed)

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data, labels, dataset = get_data()


# Experiment arguments

clf_type = 'nn'  # lr, nn

aggregation_options = 'max'  # avg, max

split_options = {
    'type': 'video',  # video, episodes
    'mean': False,  # True, False
    'config': 'IS09_emotion',  # IS09_emotion,
    'speech_embeddings': {
        'mean': False  # True, False
    }
}

nn_arch = {
    'layers': [
        ('dropout', 0.2),
        ('dense', 128, 'relu'),
        ('dropout', 0.2),
        ('dense', 64, 'tanh'),
        ('dropout', 0.20),
        ('dense', 3, 'softmax'),
    ],
    'optimizer': 'adagrad',
    'batch_size': 75,
    'epochs': 35
}

transformation_options = {
    'fulltext': 1,  # 0,1
    'numerical': 1,  # 0,1
    'nela': 1,  # 0,1
    'v_tags': 1,  # 0,1
    'open_smile': 1,  # 0,1
    'speech_embeddings': 0  # 0,1
}

evaluate_nn(data,
            labels,
            dataset,
            clf_type,
            aggregation_options=aggregation_options,
            transformation_options=transformation_options,
            split_options=split_options,
            nn_arch=nn_arch,
            debug=True,
            verbose=0)
