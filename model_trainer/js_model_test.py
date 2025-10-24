from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

def test_model():
    dim, n_classes = 8, 3
    inputs = keras.Input(shape=(dim,), name="features")
    x = layers.Dense(4, activation="relu")(inputs)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inputs, outputs)
    tfjs.converters.save_keras_model(m, "web_model_min")
    m.save("web_model_min/model.h5")

def test_sequential_model():
    dim, n_classes = 8, 3
    model = keras.Sequential([
        layers.InputLayer(input_shape=(dim,)),
        layers.Dense(4, activation="relu"),
        layers.Dense(n_classes, activation="softmax")
    ])
    tfjs.converters.save_keras_model(model, "web_model_min_seq")
    model.save("web_model_min_seq/model.keras")

if __name__ == "__main__":
    # test_model()
    test_sequential_model()

# python model_trainer/js_model_test.py
# tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model ./web_model_min web_model_min_tfjs
# tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model ./web_model_min/model.keras web_model_min_tfjs
# tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model ./web_model_min_seq/model.keras web_model_min_tfjs
