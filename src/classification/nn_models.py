"""
"""
from tensorflow import keras


def reset_weights(model):
    """
    https://github.com/keras-team/keras/issues/341
    https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
    => should not reset to weight = 0
    :param model:
    :return:
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # if you're using a model as a layer
            reset_weights(layer)  # apply function recursively
            continue

        # where are the initializers?
        if hasattr(layer, "cell"):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  # is this item an initializer?
                continue  # if no, skip it
            # find the corresponding variable, like the kernel or the bias
            if key == "recurrent_initializer":  # special case check
                var = getattr(init_container, "recurrent_kernel")
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))  # use the initializer


def shallow_NN(
    num_chans=32, num_features=256, num_hidden=16, activation="relu", learning_rate=1e-3
):
    model = keras.Sequential()
    model.add(
        keras.layers.Flatten(
            input_shape=(
                num_chans,
                num_features,
            )
        )
    )
    model.add(keras.layers.Dense(num_hidden, activation=activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
    return model


def shallow_NN2(
    num_chans=32, num_features=256, num_hidden=16, activation="relu", learning_rate=1e-3
):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(num_features,)))
    model.add(keras.layers.Dense(num_hidden, activation=activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
    return model


def DNNa_2l(num_chans=32, num_features=256, num_hidden=60, activation="relu", learning_rate=1e-3):
    model = keras.Sequential()
    model.add(
        keras.layers.Flatten(
            input_shape=(
                num_chans,
                num_features,
            )
        )
    )
    model.add(keras.layers.Dense(num_hidden, activation=activation))
    model.add(keras.layers.Dense(num_hidden // 2, activation=activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

    return model


#################################### Convolution Neural Network ####################################


def SCNNa(num_chans=32, num_features=256, learning_rate=1e-3, filters=50):
    """
    Shallow CNN
    """
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=filters,
            kernel_size=(25, 1),
            padding="same",
            activation="elu",
            input_shape=(num_chans, num_features, 1),
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(5, 1), strides=(3, 1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
    return model


# need these for ShallowConvNet
def square(x):
    return keras.backend.square(x)


def log(x):
    return keras.backend.log(keras.backend.clip(x, min_value=1e-7, max_value=10000))


def SCNNb(
    num_chans=32,
    num_features=256,
    learning_rate=1e-3,
    dropout_rate=0.5,
    filters=40,
    kernel_size=13,
    pool_size=75,
    strides=15,
):
    """
    Structure from Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping. Original code :
    https://github.com/vlawhern/arl-eegmodels
    """
    # start the model
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters,
            (1, kernel_size),
            input_shape=(num_chans, num_features, 1),
            padding="same",
            kernel_constraint=keras.constraints.max_norm(2.0, axis=(0, 1, 2)),
        )
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            (num_chans, 1),
            use_bias=False,
            kernel_constraint=keras.constraints.max_norm(2.0, axis=(0, 1, 2)),
        )
    )
    model.add(keras.layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
    model.add(keras.layers.Activation(square))
    model.add(keras.layers.AveragePooling2D(pool_size=(1, pool_size), strides=(1, strides)))
    model.add(keras.layers.Activation(log))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            1, kernel_constraint=keras.constraints.max_norm(0.5), activation="sigmoid"
        )
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])
    return model


def custom_eegnet(
    num_chans=32,
    num_features=256,
    sampling_rate=128,
    dropout_rate=0.5,
    F1=8,
    D=2,
    learning_rate=1e-3,
):
    """
    :param num_chans: Number of channels
    :param num_features: Number of time points
    :param dropout_rate: Dropout fraction
    :param F1: Number of temporal filters (F1) to learn.
               Number of pointwise filters F2 = F1 * D.
    :param D: Number of spatial filters to learn within each temporal convolution.
    :param learning_rate: Learning rate.
    :return: model
    """
    model = keras.Sequential()
    kern_length = int(sampling_rate / 2)
    model.add(
        keras.layers.Conv2D(
            F1,
            (1, kern_length),
            padding="same",
            input_shape=(num_chans, num_features, 1),
            use_bias=False,
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.DepthwiseConv2D(
            (num_chans, 1),
            padding="valid",
            depth_multiplier=D,
            depthwise_constraint=keras.constraints.unit_norm(),
            use_bias=False,
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    P1 = int(kern_length / 16)
    model.add(keras.layers.AveragePooling2D(pool_size=(1, P1), padding="valid"))
    model.add(keras.layers.Dropout(dropout_rate))
    F2 = F1 * D
    model.add(
        keras.layers.SeparableConv2D(F2, (1, int(kern_length / 2)), use_bias=False, padding="same")
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    P2 = int(P1 * 2)
    model.add(keras.layers.AveragePooling2D(pool_size=(1, P2)))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

    return model


def eegnet(
    Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, learning_rate=1e-3
):
    """
    EEGNet inspired by this code:
    https://github.com/vlawhern/arl-eegmodels which is the Keras
    implementation of : http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            F1, (1, kernLength), padding="same", input_shape=(Chans, Samples, 1), use_bias=False
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.DepthwiseConv2D(
            (Chans, 1),
            use_bias=False,
            depth_multiplier=D,
            depthwise_constraint=keras.constraints.max_norm(1.0),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.AveragePooling2D((1, 4)))
    model.add(keras.layers.Dropout(dropoutRate))
    model.add(keras.layers.SeparableConv2D(F2, (1, 16), use_bias=False, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
    model.add(keras.layers.AveragePooling2D((1, 8)))
    model.add(keras.layers.Dropout(dropoutRate))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(1))
    model.add(keras.layers.Dense(1, kernel_constraint=keras.constraints.max_norm(0.25)))
    model.add(keras.layers.Activation("sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

    return model


#
# def DeepConvNet(nb_classes, Chans=64, Samples=256,
#                 dropoutRate=0.5):
#     """
#     Model author: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
#     """
#
#     # start the model
#     input_main = Input((Chans, Samples, 1))
#     block1 = Conv2D(25, (1, 5),
#                     input_shape=(Chans, Samples, 1),
#                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
#     block1 = Conv2D(25, (Chans, 1),
#                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
#     block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
#     block1 = Activation('elu')(block1)
#     block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
#     block1 = Dropout(dropoutRate)(block1)
#
#     block2 = Conv2D(50, (1, 5),
#                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
#     block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
#     block2 = Activation('elu')(block2)
#     block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
#     block2 = Dropout(dropoutRate)(block2)
#
#     block3 = Conv2D(100, (1, 5),
#                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
#     block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
#     block3 = Activation('elu')(block3)
#     block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
#     block3 = Dropout(dropoutRate)(block3)
#
#     block4 = Conv2D(200, (1, 5),
#                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
#     block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
#     block4 = Activation('elu')(block4)
#     block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
#     block4 = Dropout(dropoutRate)(block4)
#
#     flatten = Flatten()(block4)
#
#     dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
#     softmax = Activation('softmax')(dense)
#
#     return Model(inputs=input_main, outputs=softmax)
