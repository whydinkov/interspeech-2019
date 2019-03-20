from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD, Adam, RMSprop


def _create_model(input_dim, hl_activation, hl_dropdown_rate,
                  hl_output_dim, optimizer):
    model = Sequential()
    model.add(Dense(hl_output_dim,
                    activation=hl_activation,
                    input_dim=input_dim))
    model.add(Dropout(hl_dropdown_rate))
    model.add(Dense(3, activation='softmax'))

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return parallel_model


def create_nn_clf(input_dim, args, verbose):
    return KerasClassifier(build_fn=_create_model,
                           verbose=verbose,
                           input_dim=input_dim, **args)
