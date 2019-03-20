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

_os_pipe = ('open_smile', Pipeline([
    ('selector', ColumnSelector(columns='emo_1')),
    ('to_list', FunctionTransformer(lambda X: X.tolist(), validate=False)),
    ('norm', Normalizer())
]))


def create_transfomer(include_open_smile=True):
    pipelines = [
        _fulltext_pipeline,
        _numerical_pipeline,
        _nela_desc_pipeline,
        _v_tags_pipeline
    ]

    if (include_open_smile):
        pipelines.append(_os_pipe)

    return FeatureUnion(pipelines)
