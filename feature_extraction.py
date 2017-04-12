import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.models import Model
from keras.layers import Input, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg_traffic_100_bottleneck_features_train.p', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'vgg_traffic_bottleneck_features_validation.p', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    classes = len(np.unique(y_train))
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inp, x)

    # TODO: train your model here
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    
    y_one_hot = label_binarizer.fit_transform(y_train)

    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
