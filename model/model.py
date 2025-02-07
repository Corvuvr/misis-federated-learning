import tensorflow as tf

# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Model:
    def __init__(self, learning_rate, classes_, alpha_: float = 1.0, scale_input: int = 1):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        input_shape = [32, 32, 3]
        input_shape[0] *= scale_input
        input_shape[1] *= scale_input
        input_shape=tuple(input_shape)

        print(f'{input_shape}')
        self.model = tf.keras.applications.MobileNetV2(
            input_shape, classes=classes_, alpha=alpha_, weights=None
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model
