import tensorflow as tf

from framework.base_trainer import BaseTrainer
class GPUTrainer(BaseTrainer):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config)
        self.strategy = tf.distribute.MirroredStrategy()
        
    def prepare_data(self):
        # Using tf.data.Dataset for better GPU utilization
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(10000) \
            .batch(self.model_config.get('batch_size', 32))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(self.model_config.get('batch_size', 32))
            
        return train_dataset, test_dataset
        
    def build_model(self):
        with self.strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        return model
        
    def train(self):
        train_dataset, test_dataset = self.prepare_data()
        model = self.build_model()
        
        history = model.fit(
            train_dataset,
            epochs=self.model_config.get('epochs', 5),
            validation_data=test_dataset
        )
        
        self.save_model(model, self.model_config['save_path'])
        return history