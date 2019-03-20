from evaluation import evaluate_nn
from data_retrieval import get_data

data, labels = get_data()

# Experiment options

clf_type = 'lr'

aggregation_options = 'max'

split_options = {
    'type': 'video',
    'config': 'IS09_emotion',
}

nn_arch = [
    ('dense', 5, 'tanh', 0.2),
    ('dense', 5, 'tanh', 0.2)
    ('dense', 5, 'tanh', 0.2)
]

transformation_options = {
    'fulltext': True,
    'numerical': True,
    'nela': True,
    'v_tags': True,
    'open_smile': True
}

evaluate_nn(data,
            labels,
            clf_type,
            aggregation_options=aggregation_options,
            transfomration_options=transformation_options,
            split_options=split_options,
            nn_arch=nn_arch,
            debug=True,
            verbose=1)
