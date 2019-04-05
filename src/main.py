import sys
import os
import argparse
import pickle
from datetime import datetime
from os import environ
from os.path import join
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from keras import backend as K
from tensorflow import set_random_seed
from numpy.random import seed
from evaluation.evaluation import evaluate_nn
from data_retrieval.data_retrieval import get_data

# Environment variables
load_dotenv()
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Verbosity level is too high by default

# Set predifined random seeds for both numpy and tensorflow
# to get reproduceable experiment results.
# Comment next 4 lines, to remove predefined random seeds.
np_seed = 61619
tf_seed = 25383
seed(np_seed)
set_random_seed(tf_seed)

# Input arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    '--split',
    help=('Possible values: "video", "episodes", "both".'
          'Used to split channel to classifiable parts.'),
    default='video'
)

parser.add_argument(
    '--agg',
    help=('Possible values: "max", "avg", "both".'
          'Used to aggregate classified values to channel.'),
    default='avg'
)

parser.add_argument(
    '--single',
    help='Six bit values, split by "," corelated to transformation_options'
)


# Load dataset
data, labels, dataset = get_data()

# Experiment arguments
# Possible values are visible in comments after properties
clf_type = 'nn'  # lr, nn

split_options = {
    'type': 'video',  # video, episodes
    'mean': False,  # True, False
    'config': 'IS09_emotion',  # IS09_emotion, IS12_speaker_trait
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
    'bert_fulltext': 1,  # 0,1
    'numerical': 1,  # 0,1
    'nela_desc': 1,  # 0,1
    'bert_subs': 1,  # 0,1
    'open_smile': 1,  # 0,1
    'speech_embeddings': 1,  # 0,1
}


experiment_setups = [
    [1, 0, 0, 0, 0, 0],  # only bert_fulltext
    [0, 1, 0, 0, 0, 0],  # only numerical
    [0, 0, 1, 0, 0, 0],  # only nela_desc
    [0, 0, 0, 1, 0, 0],  # only bert_subs
    [0, 0, 0, 0, 1, 0],  # only open_smile(emo_1)
    [0, 0, 0, 0, 0, 1],  # only speech_embeddings
    [1, 1, 1, 1, 0, 0],  # text+numeric without sound
    [1, 1, 1, 1, 1, 1],  # all possible together
    [1, 1, 1, 1, 1, 0],  # text+numeric+open_smile
    [1, 1, 1, 1, 0, 1],  # text+numeric+speech_embeddings
    [1, 1, 1, 0, 1, 1],  # all without bert_subs
    [1, 1, 0, 1, 1, 1],  # all without nela_desc
    [1, 0, 1, 1, 1, 1],  # all without numerical
    [0, 1, 1, 1, 1, 1],  # all without bert_fulltext
]

input_args = vars(parser.parse_args())


def parse_transformation(input):
    number = int(input)

    if number not in (0, 1):
        raise Exception("Possible values for transformation: (0, 1)")

    return number


if input_args['single']:
    experiment_setup = [parse_transformation(x)
                        for x in input_args['single'].split(',')]

    if len(experiment_setup) != 6:
        raise Exception("transformation_options has mandatory length: 6")

    experiment_setups = [experiment_setup]

if input_args['agg'] == 'both':
    possible_aggregation_options = ['avg', 'max']
else:
    possible_aggregation_options = [input_args['agg']]

if input_args['split'] == 'both':
    split_types = ['video', 'episodes']
else:
    split_types = [input_args['split']]

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = join(environ['experiments_output'], current_time)
if not os.path.exists(output_path):
    os.makedirs(output_path)

experiment_results = []
for split_type in split_types:
    for aggregation_option in possible_aggregation_options:
        for experiment_setup in experiment_setups:
            sys.stdout = sys.__stdout__  # default print to console

            K.clear_session()

            current_experiment_label = (f'{datetime.now():%H:%M:%S}, '
                                        f'split: {split_type}, '
                                        f'agg: {aggregation_option}, '
                                        f'feats: {experiment_setup}')

            print(current_experiment_label)

            file_path = (f'{split_type}_{aggregation_option}_'
                         f'{"".join([str(x) for x in experiment_setup])}'
                         '.log')

            sys.stdout = open(join(output_path, file_path), 'w')
            split_options['type'] = split_type

            transformation_options['bert_fulltext'] = experiment_setup[0]
            transformation_options['numerical'] = experiment_setup[1]
            transformation_options['nela_desc'] = experiment_setup[2]
            transformation_options['bert_subs'] = experiment_setup[3]
            transformation_options['open_smile'] = experiment_setup[4]
            transformation_options['speech_embeddings'] = experiment_setup[5]

            print(f'tensorflow.set_random_seed {tf_seed}')
            print(f'numpy.randomseed {seed}')

            result = evaluate_nn(data,
                                 labels,
                                 dataset,
                                 clf_type,
                                 aggregation_options=aggregation_option,
                                 transformation_options=transformation_options,
                                 split_options=split_options,
                                 nn_arch=nn_arch,
                                 debug=True,
                                 verbose=0)

            acc_channels_test = np.average(result[0])
            acc_channels_train = np.average(result[1])
            acc_splits_test = np.average(result[2])
            acc_splits_train = np.average(result[3])
            mae_channels_test = np.average(result[4])
            mae_channels_train = np.average(result[5])
            mae_splits_test = np.average(result[6])
            mae_splits_train = np.average(result[7])

            experiment_results.append([
                f'{split_type}, {aggregation_option}, {experiment_setup}',
                acc_splits_test,
                acc_splits_train,
                acc_channels_test,
                acc_channels_train,
                mae_splits_test,
                mae_splits_train,
                mae_channels_test,
                mae_channels_train,
            ])

# just in case if df fails for encoding reasons.
with open('experiment_results.backup', 'wb') as f:
    pickle.dump(experiment_results, f)

df = pd.DataFrame(experiment_results, columns=['experiment',
                                               'acc_splits_test',
                                               'acc_splits_train',
                                               'acc_channels_test',
                                               'acc_channels_train',
                                               'mae_splits_test',
                                               'mae_splits_train',
                                               'mae_channels_test',
                                               'mae_channels_train'])

total_results_path = join(output_path, f'{current_time}.xlsx')
df.to_excel(total_results_path, index=False)
