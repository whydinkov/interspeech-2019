import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, FunctionTransformer


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            if isinstance(self.columns, str):
                raise KeyError(
                    f"The DataFrame does not include the column {self.columns}"
                )

            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(
                "The DataFrame does not include the columns: %s" % cols_error)


_fulltext_pipeline = ('fulltext', Pipeline([
    ('selector', ColumnSelector(columns='fulltext')),
    ('tfidf', TfidfVectorizer(sublinear_tf=True,
                              min_df=5,
                              norm='l2',
                              ngram_range=(1, 2),
                              stop_words='english')),
    ('dim_red', TruncatedSVD(200, random_state=0))
]))

_numerical_pipeline = ('numerical', Pipeline([
    ('selector', ColumnSelector(columns=[
        'likes',
        'dislikes',
        'comments',
        'views',
        'duration'
    ])),
    ('norm', Normalizer())
]))

_v_tags_pipeline = ('tags', Pipeline([
    ('selector', ColumnSelector(columns='tags')),
    ('tfidf', TfidfVectorizer(sublinear_tf=True,
                              min_df=5,
                              norm='l2',
                              ngram_range=(1, 2),
                              stop_words='english')),
    ('dim_red', TruncatedSVD(10, random_state=0))
]))

_nela_desc_pipeline = ('nela', Pipeline([
    ('selector', ColumnSelector(columns='nela')),
    ('vect', DictVectorizer()),
    ('norm', Normalizer())
]))

_open_smile = ('open_smile', Pipeline([
    ('selector', ColumnSelector(columns='emo_1')),
    ('to_list', FunctionTransformer(lambda X: X.tolist(), validate=False)),
    ('norm', Normalizer())
]))


def create_transfomer(transformation_options):
    if 'fulltext' not in transformation_options:
        raise Exception('TransformerOptions. Missing "fulltext".')
    if 'numerical' not in transformation_options:
        raise Exception('TransformerOptions. Missing "numerical".')
    if 'nela' not in transformation_options:
        raise Exception('TransformerOptions. Missing "nela".')
    if 'v_tags' not in transformation_options:
        raise Exception('TransformerOptions. Missing "v_tags".')
    if 'open_smile' not in transformation_options:
        raise Exception('TransformerOptions. Missing "open_smile".')

    pipelines = []

    if transformation_options['fulltext']:
        pipelines.append(_fulltext_pipeline)
    if transformation_options['numerical']:
        pipelines.append(_numerical_pipeline)
    if transformation_options['nela']:
        pipelines.append(_nela_desc_pipeline)
    if transformation_options['v_tags']:
        pipelines.append(_v_tags_pipeline)
    if transformation_options['open_smile']:
        pipelines.append(_open_smile)

    return FeatureUnion(pipelines)
