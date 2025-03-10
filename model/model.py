import tensorflow as tf
# from tensorflow.keras.layers import Input, Resizing

# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Model:
    def __init__(self, learning_rate, classes_, alpha_: float = 1.0, scale_input: int = 1):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        init_shape = (32, 32, 3)
        input_shape = list(init_shape)
        input_shape[0] *= scale_input
        input_shape[1] *= scale_input
        input_shape=tuple(input_shape)

        input_layer = tf.keras.layers.Input(shape=init_shape)
        resized_layer = tf.keras.layers.Resizing(32*scale_input, 32*scale_input)(input_layer)

        base_model = tf.keras.applications.MobileNetV2(
            input_tensor=resized_layer,
            include_top=False,
            classes=classes_, 
            alpha=alpha_, 
            weights=None
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        x = tf.keras.layers.Dense(classes_, activation='softmax')(x)
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model
