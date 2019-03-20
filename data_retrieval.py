import pickle
import pandas as pd
from dotenv import load_dotenv
from os import environ

load_dotenv()


def get_dataset():
    if 'dataset' not in environ:
        raise Exception('Expecting dataset env.variable towards dataset path.')

    db_channels_path = environ['dataset']

    with open(db_channels_path, 'rb') as f:
        return pickle.load(f)


def get_data():
    db_channels = get_dataset()

    channels_bias = []
    for c in db_channels:
        channels_bias.append(
            [c['youtube_id'], c['bias'].replace('extreme', '')])

    channels_bias_df = pd.DataFrame(
        channels_bias, columns=['youtube_id', 'bias'])

    data = channels_bias_df['youtube_id']
    labels = channels_bias_df['bias']

    dataset = db_channels

    return (data, labels, db_channels)
