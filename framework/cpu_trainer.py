import tensorflow as tf

from framework.base_trainer import BaseTrainer
class CPUTrainer(BaseTrainer):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config)
        
    def prepare_data(self):
        # Simple MNIST data loading for demonstration
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return (x_train, y_train), (x_test, y_test)
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
        
    def train(self):
        (x_train, y_train), (x_test, y_test) = self.prepare_data()
        model = self.build_model()
        
        history = model.fit(
            x_train, y_train,
            epochs=self.model_config.get('epochs', 5),
            batch_size=self.model_config.get('batch_size', 32),
            validation_data=(x_test, y_test)
        )
        
        self.save_model(model, self.model_config['save_path'])
        return history