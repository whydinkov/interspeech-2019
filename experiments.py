from evaluation import evaluate_nn
from data_retrieval import get_data
from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data, labels, dataset = get_data()


# Experiment options

clf_type = 'nn'

aggregation_options = 'avg'

split_options = {
    'type': 'video',
    'mean': False,
    'config': 'IS09_emotion',
}

nn_arch = {
    'layers': [
        ('dropout', 0.2),
        ('dense', 32, 'tanh'),
        ('dropout', 0.2),
        ('dense', 16, 'tanh'),
        ('dropout', 0.2),
        ('dense', 3, 'softmax'),
    ],
    'optimizer': 'adam',
    'batch_size': 75,
    'epochs': 50
}

transformation_options = {
    'fulltext': True,
    'numerical': True,
    'nela': True,
    'v_tags': True,
    'open_smile': True
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
            verbose=1)
