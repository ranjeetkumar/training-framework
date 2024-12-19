from abc import ABC, abstractmethod
import os
import tensorflow as tf
import logging

class BaseTrainer(ABC):
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def prepare_data(self):
        pass
        
    @abstractmethod
    def build_model(self):
        pass
        
    @abstractmethod
    def train(self):
        pass
        
    def save_model(self, model, path):
        try:
            model.save(path)
            self.logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")