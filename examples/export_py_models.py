import json
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(43)

from keras2pmml import keras2pmml
from sklearn.datasets import make_regression, make_classification
from sklearn.neural_network import MLPClassifier, MLPRegressor
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K


layer_sizes = (10,5)
activation = ["relu", "logistic", "tanh"]
n_sample_preds = 5
base_filename = "examples/data/"


# require high precision for predictions with Keras mode
K.set_floatx("float64")


def export_mlp(filename, mlp, sample_input, activation_fct, last_activation, predict_proba=False):
    sample_preds = mlp.predict_proba(sample_input) if predict_proba else mlp.predict(sample_input)
    data = {
        "weights": dict((str(i), w.tolist()) for i,w in enumerate(mlp.coefs_)),
        "biases": dict((str(i), b.tolist()) for i,b in enumerate(mlp.intercepts_)),
        "activations": dict([(str(i), activation_fct) for i in range(len(layer_sizes))] + [(str(len(layer_sizes)), last_activation)]),
        "samples": {
            "X": sample_input.tolist(),
            "y": sample_preds.tolist()
        }
    }
    with(open(filename, "w")) as f:
        json.dump(data, f)
    print("Saved model to {}".format(filename))


def export_regressor(mlp, activation_fct, sample_input):
    fn = "{}mlp_{}_{}.json".format(base_filename, activation_fct, "regressor")
    export_mlp(fn, mlp, sample_input, activation_fct, "identity")


def export_classifier(mlp, activation_fct, sample_input, multi_label=False):
    last_activation = "sigmoid" if (mlp.n_outputs_ == 1 or multi_label) else "softmax"
    fn = "{}mlp_{}_{}.json".format(base_filename, activation_fct, "classifier")
    export_mlp(fn, mlp, sample_input, activation_fct, last_activation, predict_proba=True)


def export_keras(model, activation_fct, sample_input, model_type):
    fn = "{}keras_{}_{}".format(base_filename, activation_fct, model_type)
    sample_preds = model.predict(sample_input)
    if model_type == "regressor":
        sample_preds = sample_preds.flatten()
    sample_data = {
        "samples" : {
            "X": sample_input.tolist(),
            "y": sample_preds.tolist()
        }
    }
    keras2pmml(estimator=model, file=fn + ".pmml")
    with(open(fn + ".json", "w")) as f:
        json.dump(sample_data, f)
    print("Saved model to {}.pmml/json".format(fn))


### Regression 

Xr, yr = make_regression(n_samples=100, n_features=5, n_informative=3, n_targets=1, bias=1.0, noise=0.1, random_state=0)

for act in activation:

    # sklearn MLP
    regressor = MLPRegressor(
        hidden_layer_sizes=layer_sizes,
        activation=act,
        learning_rate_init=0.01,
        max_iter=1000,
        verbose=0,
        random_state=1
    )
    regressor.fit(Xr, yr)
    export_regressor(regressor, act, Xr[:n_sample_preds, :])

    # keras
    if act == "logistic":
        act = "sigmoid"
    model = Sequential()
    model.add(Dense(input_dim=Xr.shape[1], units=layer_sizes[0], activation=act))
    model.add(Dense(units=layer_sizes[1], activation=act))
    model.add(Dense(units=1, activation="linear"))
    model.compile(SGD(lr=0.0005), loss='mse')
    model.fit(Xr, yr, epochs=100, batch_size=16, verbose=0, validation_split=0.1, shuffle=False)
    export_keras(model, act, Xr[:n_sample_preds, :], "regressor")


### Classification

Xc, yc = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, flip_y=0.1, random_state=0, n_clusters_per_class=1)
cat_yc = to_categorical(yc, num_classes=None)

for act in activation:

    # sklearn MLP
    classifier = MLPClassifier(
        hidden_layer_sizes=layer_sizes,
        activation=act,
        learning_rate_init=0.01,
        max_iter=500,
        verbose=0,
        random_state=1
    )
    classifier.fit(Xc, yc)
    export_classifier(classifier, act, Xc[:n_sample_preds, :])

    # keras
    if act == "logistic":
        act = "sigmoid"
    model = Sequential()
    model.add(Dense(input_dim=Xc.shape[1], units=layer_sizes[0], activation=act))
    model.add(Dense(units=layer_sizes[1], activation=act))
    model.add(Dense(units=cat_yc.shape[1], activation="softmax"))
    model.compile(SGD(lr=0.01), loss='categorical_crossentropy')
    model.fit(Xc, cat_yc, epochs=100, batch_size=16, verbose=0, validation_split=0.1, shuffle=False)
    export_keras(model, act, Xc[:n_sample_preds, :], "classifier")
