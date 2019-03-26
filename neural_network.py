from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout


def _generate_keras_layer(layer, input_dim=None):
    layer_type = layer[0]

    if layer_type == 'dropout' and input_dim:
        _, dropout_rate = layer
        return Dropout(dropout_rate, input_shape=(input_dim,))

    if layer_type == 'dense' and input_dim:
        _, output_dim, activation = layer
        return Dense(output_dim, activation=activation, input_dim=input_dim)

    if layer_type == 'dropout':
        _, dropout_rate = layer
        return Dropout(dropout_rate)

    _, output_dim, activation = layer
    return Dense(output_dim, activation=activation)


def _create_model(input_dim, layers, optimizer):
    model = Sequential()

    for index, layer in enumerate(layers):
        if index == 0:  # first layer, should have input_dim
            model.add(_generate_keras_layer(layer, input_dim))

        model.add(_generate_keras_layer(layer))

    # parallel_model = multi_gpu_model(model, gpus=2)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return parallel_model


def create_nn_clf(input_dim, nn_arch, verbose):
    return KerasClassifier(build_fn=_create_model,
                           verbose=verbose,
                           input_dim=input_dim,
                           **nn_arch)
