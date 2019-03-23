from sklearn.linear_model import LogisticRegression


def create_clf():
    return LogisticRegression(multi_class='auto',
                              solver='liblinear',
                              tol=0.02,
                              C=4.5,
                              penalty='l1')
