import sys
import pprint
import numpy as np
from datetime import datetime
from preprocessing.aggregate import get_channels_bias_avg
from preprocessing.aggregate import get_channels_bias_max
from preprocessing.aggregate import get_labels_from_proba
from models.neural_network import create_nn_clf
from models.logistic_regression import create_clf
from preprocessing.preprocessing import split_channel
from features.pipelines import create_transfomer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error
from .scoring import calc_mae

import pandas as pd
from timeit import default_timer as timer
from tabulate import tabulate
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def evaluate_nn(
    data,
    labels,
    dataset,
    clf_type,
    aggregation_options='avg',
    transformation_options={},
    split_options={},
    nn_arch={},
    debug=False,
    verbose=0
):
    pp = pprint.PrettyPrinter(indent=4, stream=sys.stdout)

    if debug:
        current_time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        print(f'Experiment: {current_time}')
        print(f'Classifier type: {clf_type}')
        print(f'Included features:')
        pp.pprint(transformation_options)
        print(f'Aggregation type: {aggregation_options}')
        print(f'Split options:')
        pp.pprint(split_options)
        if clf_type == 'nn':
            print(f'Neural network architecture:')
            pp.pprint(nn_arch)
        log_line()

    splits_test_scores = []
    splits_train_scores = []
    channels_test_scores = []
    channels_train_scores = []
    experiments_times = []
    f1_micro_split_test_scores = []
    f1_micro_split_train_scores = []
    f1_macro_split_test_scores = []
    f1_macro_split_train_scores = []
    f1_micro_channels_test_scores = []
    f1_micro_channels_train_scores = []
    f1_macro_channels_test_scores = []
    f1_macro_channels_train_scores = []
    mae_split_test_scores = []
    mae_split_train_scores = []
    mae_channels_test_scores = []
    mae_channels_train_scores = []

    split_type = split_options['type']

    skf = StratifiedKFold(n_splits=5)
    for index, (train_index, test_index) in enumerate(skf.split(data, labels)):
        start = timer()

        # split
        X_train_channels = data.iloc[train_index]
        X_test_channels = data.iloc[test_index]
        y_train_channels = labels.iloc[train_index]
        y_test_channels = labels.iloc[test_index]

        # transform to features
        transformer_train = create_transfomer(transformation_options)
        X_train_splits = split_channel(
            X_train_channels, dataset, split_options)
        X_train = transformer_train.fit_transform(
            X_train_splits, X_train_splits['bias'].tolist())

        transformer_test = create_transfomer(transformation_options)
        X_test_splits = split_channel(
            X_test_channels, dataset, split_options)
        X_test = transformer_test.transform(X_test_splits)

        # clf
        input_dim = X_test.shape[1]

        if debug:
            print(f'Fold {index + 1}')
            print(f'Shape test: {X_test.shape} | train {X_train.shape}')
            print()
            print(f'Distribution:')
            dist_df_test_channels = pd.concat([
                y_test_channels.value_counts(),
                y_test_channels.value_counts(normalize=True)
            ], axis=1)
            dist_df_test_channels.columns = ['count', 'distribution']
            dist_df_test_channels.columns.name = 'channels test'
            log_df(dist_df_test_channels)
            print()

            dist_df_train_channels = pd.concat([
                y_train_channels.value_counts(),
                y_train_channels.value_counts(normalize=True)
            ], axis=1)
            dist_df_train_channels.columns = ['count', 'distribution']
            dist_df_train_channels.columns.name = 'channels train'
            log_df(dist_df_train_channels)
            print()

            dist_df_test_splits = pd.concat([
                X_test_splits['bias'].value_counts(),
                X_test_splits['bias'].value_counts(normalize=True)
            ], axis=1)
            dist_df_test_splits.columns = ['count', 'distribution']
            dist_df_test_splits.columns.name = f'{split_type} test'
            log_df(dist_df_test_splits)
            print()

            dist_df_train_splits = pd.concat([
                X_train_splits['bias'].value_counts(),
                X_train_splits['bias'].value_counts(normalize=True)
            ], axis=1)
            dist_df_train_splits.columns = ['count', 'distribution']
            dist_df_train_splits.columns.name = f'{split_type} train'
            log_df(dist_df_train_splits)
            print()

        if clf_type == 'lr':
            clf = create_clf()
        elif clf_type == 'nn':
            clf = create_nn_clf(input_dim, nn_arch, verbose)

        clf.fit(X_train, X_train_splits['bias'])

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
            X_train_splits['channel_id'].tolist(), y_pred_train_proba)

        # metrics
        videos_test_acc = accuracy_score(
            y_true=X_test_splits['bias'].tolist(), y_pred=y_pred)
        videos_train_acc = accuracy_score(
            y_true=X_train_splits['bias'].tolist(), y_pred=y_pred_train)
        channels_test_acc = accuracy_score(
            y_true=y_test_channels, y_pred=y_pred_channels)
        channels_train_acc = accuracy_score(
            y_true=y_train_channels, y_pred=y_pred_train_channels)

        splits_test_scores.append(videos_test_acc)
        splits_train_scores.append(videos_train_acc)

        channels_test_scores.append(channels_test_acc)
        channels_train_scores.append(channels_train_acc)

        end = timer()

        experiments_times.append(end-start)

        if debug:
            print()
            print()
            print('Accuracy:')
            log_fold_results(split_type, videos_test_acc, videos_train_acc)
            log_fold_results(
                'channels', channels_test_acc, channels_train_acc)
            print()

            # f1
            f1_micro_split_test = f1_score(
                y_true=X_test_splits['bias'].tolist(),
                y_pred=y_pred,
                labels=clf.classes_,
                average='micro'
            )
            f1_micro_split_test_scores.append(f1_micro_split_test)
            f1_micro_split_train = f1_score(
                y_true=X_train_splits['bias'].tolist(),
                y_pred=y_pred_train,
                labels=clf.classes_,
                average='micro',

            )
            f1_micro_split_train_scores.append(f1_micro_split_train)
            f1_macro_split_test = f1_score(
                y_true=X_test_splits['bias'].tolist(),
                y_pred=y_pred,
                labels=clf.classes_,
                average='macro'
            )
            f1_macro_split_test_scores.append(f1_macro_split_test)
            f1_macro_split_train = f1_score(
                y_true=X_train_splits['bias'].tolist(),
                y_pred=y_pred_train,
                labels=clf.classes_,
                average='macro'
            )
            f1_macro_split_train_scores.append(f1_macro_split_train)
            f1_micro_channels_test = f1_score(
                y_true=y_test_channels,
                y_pred=y_pred_channels,
                labels=clf.classes_,
                average='micro'
            )
            f1_micro_channels_test_scores.append(f1_micro_channels_test)
            f1_micro_channels_train = f1_score(
                y_true=y_train_channels,
                y_pred=y_pred_train_channels,
                labels=clf.classes_,
                average='micro'
            )
            f1_micro_channels_train_scores.append(f1_micro_channels_train)
            f1_macro_channels_test = f1_score(
                y_true=y_test_channels,
                y_pred=y_pred_channels,
                labels=clf.classes_,
                average='macro'
            )
            f1_macro_channels_test_scores.append(f1_macro_channels_test)
            f1_macro_channels_train = f1_score(
                y_true=y_train_channels,
                y_pred=y_pred_train_channels,
                labels=clf.classes_,
                average='macro'
            )
            f1_macro_channels_train_scores.append(f1_macro_channels_train)
            print()
            print('F1:')
            print(f'f1 micro {split_type} test:', f1_micro_split_test)
            print(f'f1 micro {split_type} train:', f1_micro_split_train)
            print(f'f1 macro {split_type} test:', f1_macro_split_test)
            print(f'f1 macro {split_type} train:', f1_macro_split_train)

            print(f'f1 micro channels test:', f1_micro_channels_test)
            print(f'f1 micro channels train:', f1_micro_channels_train)
            print(f'f1 macro channels test:', f1_macro_channels_test)
            print(f'f1 macro channels train:', f1_macro_channels_train)
            print()

            # Mean absolute error
            print('MAE: ')
            mae_split_test = calc_mae(y_true=X_test_splits['bias'].tolist(),
                                      y_pred=y_pred)
            mae_split_test_scores.append(mae_split_test)
            mae_split_train = calc_mae(y_true=X_train_splits['bias'].tolist(),
                                       y_pred=y_pred_train)
            mae_split_train_scores.append(mae_split_train)
            mae_channels_test = calc_mae(y_true=y_test_channels,
                                         y_pred=y_pred_channels)
            mae_channels_test_scores.append(mae_channels_test)
            mae_channels_train = calc_mae(y_true=y_train_channels,
                                          y_pred=y_pred_train_channels)
            mae_channels_train_scores.append(mae_channels_train)
            print(f'Mean absolute error ({split_type} test): ', mae_split_test)
            print(
                f'Mean absolute error ({split_type} train): ', mae_split_train)
            print(
                f'Mean absolute error (channels train): ', mae_channels_test)
            print(
                f'Mean absolute error (channels train): ', mae_channels_train)
            print(f'Done with fold: {index + 1} for {end-start:.2f}s')
            print()

            # Confusion matrixes
            print(f'Confusion matrices:')
            cm_channels = make_cm(y_true=y_test_channels,
                                  y_pred=y_pred_channels,
                                  labels=clf.classes_,
                                  cm_type='channels')
            log_df(cm_channels)
            print()
            cm_splits = make_cm(y_true=X_test_splits['bias'].tolist(),
                                y_pred=y_pred,
                                labels=clf.classes_,
                                cm_type=split_type)
            log_df(cm_splits)
            print()
            log_line()

    if debug:
        print()
        print()
        print('Experiment results:')
        print('Accuracy:')
        log_results(f'{split_type} test', splits_test_scores)
        log_results(f'{split_type} train', splits_train_scores)
        log_results('channels test', channels_test_scores)
        log_results('channels train', channels_train_scores)
        print('F1:')
        log_results(f'F1 micro {split_type} test', f1_micro_split_test_scores)
        log_results(f'F1 micro {split_type} train',
                    f1_micro_split_train_scores)
        log_results(f'F1 macro {split_type} test', f1_macro_split_test_scores)
        log_results(f'F1 macro {split_type} train',
                    f1_macro_split_train_scores)
        log_results(f'F1 micro channels test', f1_micro_channels_test_scores)
        log_results(f'F1 micro channels train', f1_micro_channels_train_scores)
        log_results(f'F1 macro channels test', f1_macro_channels_test_scores)
        log_results(f'F1 macro channels train', f1_macro_channels_train_scores)
        print('MAE:')
        log_results(f'Mean absolute error {split_type} test',
                    mae_split_test_scores)
        log_results(f'Mean absolute error {split_type} train',
                    mae_split_train_scores)
        log_results(f'Mean absolute error channels test',
                    mae_channels_test_scores)
        log_results(f'Mean absolute error channels train',
                    mae_channels_train_scores)
    return (channels_test_scores,
            channels_train_scores,
            splits_test_scores,
            splits_train_scores,
            mae_split_test_scores,
            mae_split_train_scores,
            mae_channels_test_scores,
            mae_channels_train_scores,
            experiments_times,
            nn_arch)


def log_df(df):
    df_headers = [df.columns.name] + df.columns.tolist()
    print(tabulate(df, tablefmt='grid', headers=df_headers))
    print()


def make_cm(y_true, y_pred, labels, cm_type):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    df = pd.DataFrame(cm, columns=labels, index=labels)
    df.columns.name = cm_type
    return df


def log_line():
    print('-' * 75)


def log_fold_results(result_type, test, train):
    print(f'{result_type} | test: {test:.6f} / train: {train:.6f}')


def log_results(result_type, scores):
    fold_scores = ["%.6f" % v for v in scores]
    print(f'{result_type} | {np.average(scores)}, folds: {fold_scores}')
