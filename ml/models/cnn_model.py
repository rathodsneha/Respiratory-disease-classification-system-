"""
CNN Model for Respiratory Disease Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RespiratoryCNN:
    """
    Convolutional Neural Network for respiratory disease classification
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (128, 128, 1),
                 num_classes: int = 6,
                 learning_rate: float = 0.001,
                 dropout_rate: float = 0.5):
        """
        Initialize CNN model
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of disease classes
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        # Disease class names
        self.class_names = [
            'healthy',
            'asthma', 
            'pneumonia',
            'bronchitis',
            'copd',
            'pleural_effusion'
        ]
        
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"CNN model built with input shape: {self.input_shape}")
        logger.info(f"Model parameters: {self.model.count_params():,}")
    
    def train(self, 
              train_data: np.ndarray,
              train_labels: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray],
              epochs: int = 100,
              batch_size: int = 32,
              callbacks: list = None) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            train_data: Training data
            train_labels: Training labels (one-hot encoded)
            validation_data: Validation data tuple (data, labels)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call _build_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_cnn_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        logger.info("Starting CNN model training...")
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Training labels shape: {train_labels.shape}")
        
        # Train model
        self.history = self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("CNN model training completed!")
        return self.history.history
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            data: Input data for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Get predictions
        probabilities = self.model.predict(data)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def predict_single(self, data: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Make prediction on a single sample
        
        Args:
            data: Single input sample
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        
        # Add batch dimension if needed
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=0)
        
        predictions, probabilities = self.predict(data)
        
        return predictions[0], probabilities[0]
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test data
            test_labels: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Evaluate model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_data, test_labels, verbose=0
        )
        
        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        # Get predictions for confusion matrix
        predictions, probabilities = self.predict(test_data)
        true_labels = np.argmax(test_labels, axis=1)
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from: {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        
        if self.model is None:
            return "Model not built."
        
        # Capture model summary
        from io import StringIO
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        summary = summary_io.getvalue()
        summary_io.close()
        
        return summary
    
    def get_feature_maps(self, data: np.ndarray, layer_name: str = None) -> np.ndarray:
        """
        Extract feature maps from a specific layer
        
        Args:
            data: Input data
            layer_name: Name of the layer to extract features from
            
        Returns:
            Feature maps from the specified layer
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # If no layer specified, use the last convolutional layer
        if layer_name is None:
            layer_name = 'conv2d_3'  # Last conv layer
        
        # Create feature extraction model
        feature_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Extract features
        feature_maps = feature_model.predict(data)
        
        return feature_maps
    
    def visualize_feature_maps(self, data: np.ndarray, layer_name: str = None, 
                              num_features: int = 16):
        """
        Visualize feature maps from a specific layer
        
        Args:
            data: Input data
            layer_name: Name of the layer to visualize
            num_features: Number of feature maps to display
        """
        
        feature_maps = self.get_feature_maps(data, layer_name)
        
        # Select first sample and limit number of features
        feature_maps = feature_maps[0, :, :, :num_features]
        
        # Create visualization grid
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_features):
            axes[i].imshow(feature_maps[:, :, i], cmap='viridis')
            axes[i].set_title(f'Feature {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_activation_map(self, data: np.ndarray, class_index: int) -> np.ndarray:
        """
        Generate Class Activation Map (CAM) for a specific class
        
        Args:
            data: Input data
            class_index: Index of the class to generate CAM for
            
        Returns:
            Class activation map
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")
        
        # Create model that outputs both predictions and last conv layer
        cam_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, last_conv_layer.output]
        )
        
        # Get predictions and feature maps
        predictions, feature_maps = cam_model.predict(data)
        
        # Get weights of the output layer for the specified class
        output_weights = self.model.layers[-1].get_weights()[0]
        class_weights = output_weights[:, class_index]
        
        # Generate CAM
        cam = np.zeros(feature_maps.shape[1:3])
        
        for i, weight in enumerate(class_weights):
            cam += weight * feature_maps[0, :, :, i]
        
        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

# Example usage
if __name__ == "__main__":
    # Create model instance
    cnn_model = RespiratoryCNN()
    
    # Print model summary
    print(cnn_model.get_model_summary())
    
    # Example training data (you would load your actual data)
    # train_data = np.random.random((1000, 128, 128, 1))
    # train_labels = np.random.randint(0, 6, (1000, 6))
    # validation_data = (np.random.random((200, 128, 128, 1)), 
    #                    np.random.randint(0, 6, (200, 6)))
    
    # Train model
    # history = cnn_model.train(train_data, train_labels, validation_data, epochs=10)
    
    # Save model
    # cnn_model.save_model('respiratory_cnn_model.h5')