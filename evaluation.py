import numpy as np

from aggregate import get_channels_bias_avg, get_channels_bias_max
from neural_network import create_nn_clf
from logistic_regression import create_clf
from preprocessing import split_channel
from pipelines import create_transfomer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from timeit import default_timer as timer


def evaluate_nn(
    data,
    labels,
    clf_type,
    aggregation_options='avg',
    transformation_options={},
    split_options={},
    nn_arch={},
    debug=False,
    verbose=0
):
    if debug:
        print('Experiment:')
        print(f'Clf type: {clf_type}')
        print(f'Aggregation options: {aggregation_options}')
        print(f'Transformation options: {transformation_options}')
        print(f'Split options: {split_options}')

        if clf_type == 'nn':
            print(f'NN arch: {nn_arch}')

    videos_test_scores = []
    videos_train_scores = []
    channels_test_scores = []
    channels_train_scores = []
    experiments_times = []

    skf = StratifiedKFold(n_splits=5)
    for index, (train_index, test_index) in enumerate(skf.split(data, labels)):
        start = timer()

        # split
        X_train_channels = data.iloc[train_index]
        X_test_channels = data.iloc[test_index]
        y_train_channels = labels.iloc[train_index],
        y_test_channels = labels.iloc[test_index]

        # transform to features
        transformer_train = create_transfomer(transformation_options)
        X_train_videos = split_channel(
            X_train_channels, split_options)
        X_train = transformer_train.fit_transform(
            X_train_videos, X_train_videos['bias'].tolist())

        transformer_test = create_transfomer(transformation_options)
        X_test_splits = split_channel(
            X_test_channels, split_options)
        X_test = transformer_test.transform(X_test_splits)

        # clf
        input_dim = X_test.shape[1]

        if debug:
            print(f'X_test shape: {X_test.shape}')
            print(f'X_train shape: {X_train.shape}')

        if clf_type == 'lr':
            clf = create_clf()
        elif clf_type == 'nn':
            clf = create_nn_clf(input_dim, nn_arch, verbose)

        clf.fit(X_train, X_train_videos['bias'])

        y_pred_proba = clf.predict_proba(X_test)
        y_pred_ = clf.predict(X_test)
        y_pred = get_labels_from_proba(y_pred_proba)

        assert accuracy_score(y_pred_, y_pred), 1.0

        if aggregation_options == 'avg':
            aggregate = get_channels_bias_avg
        if aggregation_options == 'max':
            aggregate = get_channels_bias_max

        y_pred_train_proba = clf.predict_proba(X_train)
        y_pred_train = get_labels_from_proba(y_pred_train_proba)

        y_pred_channels = aggregate(
            X_test_splits['channel_id'].tolist(), y_pred_proba)
        y_pred_train_channels = aggregate(
            X_train_videos['channel_id'].tolist(), y_pred_train_proba)

        # metrics
        videos_test_acc = accuracy_score(
            y_true=X_test_splits['bias'].tolist(), y_pred=y_pred)
        videos_train_acc = accuracy_score(
            y_true=X_train_videos['bias'].tolist(), y_pred=y_pred_train)
        channels_test_acc = accuracy_score(
            y_true=y_test_channels, y_pred=y_pred_channels)
        channels_train_acc = accuracy_score(
            y_true=y_train_channels, y_pred=y_pred_train_channels)

        videos_test_scores.append(videos_test_acc)
        videos_train_scores.append(videos_train_acc)

        channels_test_scores.append(channels_test_acc)
        channels_train_scores.append(channels_train_acc)

        end = timer()

        experiments_times.append(end-start)

        if debug:
            print_fold_results('videos', videos_test_acc, videos_train_acc)
            print_fold_results('videos', channels_test_acc, channels_train_acc)
            print(f'Done with split: {index + 1} ({end-start})')

    if debug:
        print_results('videos test', videos_test_scores)
        print_results('videos train', videos_train_scores)
        print_results('channels tests', channels_test_acc)
        print_results('channels train', channels_train_acc)

    return (channels_test_scores,
            channels_train_scores,
            videos_test_scores,
            videos_train_scores,
            experiments_times,
            nn_args)


def print_fold_results(result_type, test, train):
    print(f'{result_type} | test: {test:.6f} / train: {train:.6f}')


def print_results(result_type, scores):
    print(
        f'{result_type} | {np.average(scores)}, folds: {["%.5f" % v for v in videos_test_scores]}')
