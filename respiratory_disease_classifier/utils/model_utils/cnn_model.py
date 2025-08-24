import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import json
from datetime import datetime
import pickle

class CNNRespiratoryClassifier:
    """CNN model for respiratory disease classification from mel-spectrograms"""
    
    def __init__(self, input_shape=(128, 128, 1), num_classes=6, model_name="CNN_Respiratory"):
        """
        Initialize CNN model
        
        Args:
            input_shape (tuple): Input shape for mel-spectrograms (height, width, channels)
            num_classes (int): Number of disease classes
            model_name (str): Name of the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_names = ['Healthy', 'Asthma', 'Pneumonia', 'Bronchitis', 'COPD', 'Pleural_Effusion']
        
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build CNN architecture optimized for respiratory sound classification
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate * 0.5),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate * 0.7),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Global average pooling instead of flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        return model
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_callbacks(self, model_dir, patience=15):
        """
        Prepare training callbacks
        
        Args:
            model_dir (str): Directory to save model checkpoints
            patience (int): Early stopping patience
            
        Returns:
            list: List of callbacks
        """
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, f'{self.model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(model_dir, f'{self.model_name}_training_log.csv')
            )
        ]
        
        return callbacks_list
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_dir='models/cnn'):
        """
        Train the CNN model
        
        Args:
            X_train (np.array): Training spectrograms
            y_train (np.array): Training labels (one-hot encoded)
            X_val (np.array): Validation spectrograms
            y_val (np.array): Validation labels (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            model_dir (str): Directory to save model
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks(model_dir)
        
        # Data augmentation for training
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,  # Don't flip spectrograms
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Fit the data generator
        datagen.fit(X_train)
        
        print(f"Training {self.model_name} model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Input shape: {X_train.shape[1:]}")
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
        
        # Save training history
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, f'{self.model_name}_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.array): Test spectrograms
            y_test (np.array): Test labels (one-hot encoded)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curves for multiclass
        y_test_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Calculate ROC AUC for each class
        roc_auc = {}
        for i in range(self.num_classes):
            if len(np.unique(y_test_bin[:, i])) > 1:  # Check if class exists in test set
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[self.class_names[i]] = auc(fpr, tpr)
            else:
                roc_auc[self.class_names[i]] = 0.0
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc_scores': roc_auc,
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist(),
            'true_labels': y_true.tolist()
        }
        
        return evaluation_results
    
    def predict(self, X, return_probabilities=True):
        """
        Make predictions on new data
        
        Args:
            X (np.array): Input spectrograms
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get predictions
        y_pred_proba = self.model.predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get predicted class names
        predicted_classes = [self.class_names[i] for i in y_pred]
        
        # Get confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        results = {
            'predicted_classes': predicted_classes,
            'predicted_indices': y_pred.tolist(),
            'confidence_scores': confidence_scores.tolist()
        }
        
        if return_probabilities:
            # Create probability dictionary for each sample
            probabilities = []
            for i in range(len(X)):
                prob_dict = {self.class_names[j]: float(y_pred_proba[i][j]) for j in range(self.num_classes)}
                probabilities.append(prob_dict)
            results['class_probabilities'] = probabilities
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save model metadata
        metadata = {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model
        self.model = keras.models.load_model(filepath)
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_name = metadata.get('model_name', self.model_name)
                self.input_shape = tuple(metadata.get('input_shape', self.input_shape))
                self.num_classes = metadata.get('num_classes', self.num_classes)
                self.class_names = metadata.get('class_names', self.class_names)
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, output_path=None):
        """
        Plot training history
        
        Args:
            output_path (str): Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history['loss'], label='Training Loss', color='blue')
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        if 'precision' in self.history:
            axes[1, 0].plot(self.history['precision'], label='Training Precision', color='blue')
            axes[1, 0].plot(self.history['val_precision'], label='Validation Precision', color='red')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if 'recall' in self.history:
            axes[1, 1].plot(self.history['recall'], label='Training Recall', color='blue')
            axes[1, 1].plot(self.history['val_recall'], label='Validation Recall', color='red')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            output_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('CNN Model - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_maps(self, X, layer_name=None):
        """
        Extract feature maps from intermediate layers
        
        Args:
            X (np.array): Input spectrograms
            layer_name (str): Name of layer to extract features from
            
        Returns:
            np.array: Feature maps
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if layer_name is None:
            # Get feature maps from the last convolutional layer
            conv_layers = [layer for layer in self.model.layers if 'conv2d' in layer.name]
            if conv_layers:
                layer_name = conv_layers[-1].name
            else:
                raise ValueError("No convolutional layers found")
        
        # Create a model that outputs the feature maps
        feature_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_model.predict(X)
        
        return feature_maps
    
    def visualize_feature_maps(self, X, sample_index=0, output_path=None):
        """
        Visualize feature maps for a single sample
        
        Args:
            X (np.array): Input spectrograms
            sample_index (int): Index of sample to visualize
            output_path (str): Path to save the visualization
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get feature maps from multiple layers
        conv_layer_names = [layer.name for layer in self.model.layers if 'conv2d' in layer.name]
        
        if not conv_layer_names:
            raise ValueError("No convolutional layers found")
        
        # Select a few representative layers
        selected_layers = conv_layer_names[::2][:4]  # Every other layer, max 4
        
        fig, axes = plt.subplots(len(selected_layers), 8, figsize=(20, len(selected_layers) * 3))
        
        for i, layer_name in enumerate(selected_layers):
            feature_maps = self.get_feature_maps(X[sample_index:sample_index+1], layer_name)
            
            # Show first 8 feature maps
            for j in range(min(8, feature_maps.shape[-1])):
                ax = axes[i, j] if len(selected_layers) > 1 else axes[j]
                ax.imshow(feature_maps[0, :, :, j], cmap='viridis')
                ax.set_title(f'{layer_name}\nFeature {j+1}')
                ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()