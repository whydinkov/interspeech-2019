import numpy as np
import pandas as pd
from data_retrieval import get_dataset


def data_transformation(channels, split_options):
    result = []

    for channel in channels:
        channel_id = channel['youtube_id']
        bias = channel['bias'].replace('extreme', '')

        for video in channel['videos']:
            # fulltext
            title = video['snippet']['title']
            description = video['snippet']['description']
            tags = ' '.join(video['snippet'].get('tags', []))

            fulltext = ' '.join([title, description, tags])

            # numerical
            views = video['statistics'].get('viewCount', 0)
            likes = video['statistics'].get('likeCount', 0)
            dislikes = video['statistics'].get('dislikeCount', 0)
            comments = video['statistics'].get('commentCount', 0)
            duration = video['contentDetails']['duration_seconds']

            # nela
            nela = video['nela']['title_description']

            for feats in get_open_smile_features(
                    video['open_smile'],
                    split_options):
                result.append([channel_id,
                               fulltext,
                               tags,
                               views,
                               likes,
                               dislikes,
                               comments,
                               duration,
                               nela,
                               feats,
                               bias])

    return result


def get_open_smile_features(open_smile, options):
    if not open_smile:
        raise Exception('Missing open_smile features for video.')

    get_mean = options.get('mean', False)
    by_video = options.get('type', 'video') == 'video'
    config = options.get('config', 'IS09_emotion')

    if config not in open_smile:
        raise Exception(f'Missing config. Available: {open_smile.keys()}')

    if not by_video and get_mean:
        raise Exception('Mean is applied only for videos not speech episodes.')

    if by_video and get_mean:
        return [calculate_mean(open_smile[config])]
    elif by_video:
        return [open_smile[config]['1']]
    else:
        return open_smile[config].values()


def calculate_mean(features_dict):
    return np.average(list(features_dict.values()), axis=0).tolist()

# split options
#   'single' - [True, False] - if true return only videos, if false return
#   speach episodes.
#   'mean' - [True, False] - get mean representation of open_smile
#   features (only if single=True).
#   'config' - ['IS09_emotion', 'IS12_speaker_traints'] open_smile config


def split_channel(channel_ids, dataset, split_options):
    db_channels = dataset

    channel_ids = channel_ids.tolist()

    current_channels = [
        c for c in db_channels if c['youtube_id'] in channel_ids]

    splits = data_transformation(current_channels, split_options)

    return pd.DataFrame(splits, columns=['channel_id',
                                         'fulltext',
                                         'tags',
                                         'views',
                                         'likes',
                                         'dislikes',
                                         'comments',
                                         'duration',
                                         'nela',
                                         'open_smile',
                                         'bias'])
