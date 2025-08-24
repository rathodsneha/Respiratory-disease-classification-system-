import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler
import os
import json
from datetime import datetime
import pickle

class HybridCNNLSTMClassifier:
    """Hybrid CNN-LSTM model for respiratory disease classification"""
    
    def __init__(self, cnn_input_shape=(128, 128, 1), lstm_input_shape=(100, 39), 
                 num_classes=6, model_name="Hybrid_CNN_LSTM"):
        """
        Initialize Hybrid CNN-LSTM model
        
        Args:
            cnn_input_shape (tuple): Input shape for CNN (mel-spectrograms)
            lstm_input_shape (tuple): Input shape for LSTM (sequential features)
            num_classes (int): Number of disease classes
            model_name (str): Name of the model
        """
        self.cnn_input_shape = cnn_input_shape
        self.lstm_input_shape = lstm_input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.scaler = None
        self.class_names = ['Healthy', 'Asthma', 'Pneumonia', 'Bronchitis', 'COPD', 'Pleural_Effusion']
        
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build Hybrid CNN-LSTM architecture
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        # CNN branch for spatial feature extraction from spectrograms
        cnn_input = layers.Input(shape=self.cnn_input_shape, name='cnn_input')
        
        # CNN layers
        x_cnn = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(cnn_input)
        x_cnn = layers.BatchNormalization()(x_cnn)
        x_cnn = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(x_cnn)
        x_cnn = layers.MaxPooling2D((2, 2))(x_cnn)
        x_cnn = layers.Dropout(dropout_rate * 0.5)(x_cnn)
        
        x_cnn = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(x_cnn)
        x_cnn = layers.BatchNormalization()(x_cnn)
        x_cnn = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(x_cnn)
        x_cnn = layers.MaxPooling2D((2, 2))(x_cnn)
        x_cnn = layers.Dropout(dropout_rate * 0.7)(x_cnn)
        
        x_cnn = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(x_cnn)
        x_cnn = layers.BatchNormalization()(x_cnn)
        x_cnn = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.001))(x_cnn)
        x_cnn = layers.MaxPooling2D((2, 2))(x_cnn)
        x_cnn = layers.Dropout(dropout_rate)(x_cnn)
        
        # Global Average Pooling for CNN features
        cnn_features = layers.GlobalAveragePooling2D()(x_cnn)
        cnn_features = layers.Dense(256, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001))(cnn_features)
        cnn_features = layers.BatchNormalization()(cnn_features)
        cnn_features = layers.Dropout(dropout_rate)(cnn_features)
        
        # LSTM branch for temporal feature analysis
        lstm_input = layers.Input(shape=self.lstm_input_shape, name='lstm_input')
        
        # LSTM layers
        x_lstm = layers.LSTM(96, return_sequences=True, dropout=dropout_rate,
                            recurrent_dropout=dropout_rate * 0.5,
                            kernel_regularizer=regularizers.l2(0.001))(lstm_input)
        x_lstm = layers.BatchNormalization()(x_lstm)
        
        x_lstm = layers.LSTM(64, return_sequences=True, dropout=dropout_rate,
                            recurrent_dropout=dropout_rate * 0.5,
                            kernel_regularizer=regularizers.l2(0.001))(x_lstm)
        x_lstm = layers.BatchNormalization()(x_lstm)
        
        # Attention mechanism for LSTM
        attention = layers.Dense(1, activation='tanh')(x_lstm)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        lstm_attended = layers.Multiply()([x_lstm, attention])
        lstm_features = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(lstm_attended)
        
        lstm_features = layers.Dense(256, activation='relu',
                                    kernel_regularizer=regularizers.l2(0.001))(lstm_features)
        lstm_features = layers.BatchNormalization()(lstm_features)
        lstm_features = layers.Dropout(dropout_rate)(lstm_features)
        
        # Fusion layer - combine CNN and LSTM features
        combined = layers.Concatenate()([cnn_features, lstm_features])
        
        # Cross-attention between CNN and LSTM features
        cnn_attention = layers.Dense(256, activation='tanh')(cnn_features)
        lstm_attention = layers.Dense(256, activation='tanh')(lstm_features)
        
        # Compute attention weights
        attention_scores = layers.Multiply()([cnn_attention, lstm_attention])
        attention_weights = layers.Dense(1, activation='sigmoid')(attention_scores)
        
        # Apply attention to combined features
        attended_cnn = layers.Multiply()([cnn_features, attention_weights])
        attended_lstm = layers.Multiply()([lstm_features, 
                                          layers.Lambda(lambda x: 1 - x)(attention_weights)])
        
        # Final feature fusion
        fused_features = layers.Add()([attended_cnn, attended_lstm])
        
        # Additional processing layers
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Combine with fused features
        x = layers.Concatenate()([x, fused_features])
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.3)(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = models.Model(inputs=[cnn_input, lstm_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        return model
    
    def build_sequential_hybrid_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build Sequential Hybrid model where CNN features feed into LSTM
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        # Input for spectrograms
        input_layer = layers.Input(shape=self.cnn_input_shape)
        
        # CNN feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.7)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Reshape CNN output for LSTM input
        # Calculate the flattened size after convolutions
        cnn_output_shape = x.shape[1:]  # (height, width, channels)
        timesteps = cnn_output_shape[0] * cnn_output_shape[1]  # height * width
        features = cnn_output_shape[2]  # channels
        
        x = layers.Reshape((timesteps, features))(x)
        
        # LSTM layers for temporal analysis
        x = layers.LSTM(128, return_sequences=True, dropout=dropout_rate,
                       recurrent_dropout=dropout_rate * 0.5,
                       kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(64, return_sequences=False, dropout=dropout_rate,
                       recurrent_dropout=dropout_rate * 0.5,
                       kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = models.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        return model
    
    def prepare_dual_input_data(self, spectrograms, features_list, labels_list):
        """
        Prepare data for dual-input hybrid model
        
        Args:
            spectrograms (np.array): Mel-spectrograms for CNN
            features_list (list): Sequential features for LSTM
            labels_list (list): Labels
            
        Returns:
            tuple: (X_cnn, X_lstm, y)
        """
        # Prepare CNN input (spectrograms)
        X_cnn = spectrograms
        
        # Prepare LSTM input (sequential features)
        from .lstm_model import LSTMRespiratoryClassifier
        lstm_classifier = LSTMRespiratoryClassifier()
        X_lstm, y = lstm_classifier.prepare_sequential_data(features_list, labels_list)
        
        return X_cnn, X_lstm, y
    
    def normalize_features(self, X_lstm_train, X_lstm_val=None, X_lstm_test=None):
        """
        Normalize LSTM features using StandardScaler
        
        Args:
            X_lstm_train (np.array): Training LSTM features
            X_lstm_val (np.array): Validation LSTM features
            X_lstm_test (np.array): Test LSTM features
            
        Returns:
            tuple: Normalized LSTM features
        """
        # Reshape for scaling
        original_shape = X_lstm_train.shape
        X_train_flat = X_lstm_train.reshape(-1, X_lstm_train.shape[-1])
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        results = [X_train_scaled]
        
        # Transform validation data
        if X_lstm_val is not None:
            X_val_flat = X_lstm_val.reshape(-1, X_lstm_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled.reshape(X_lstm_val.shape)
            results.append(X_val_scaled)
        
        # Transform test data
        if X_lstm_test is not None:
            X_test_flat = X_lstm_test.reshape(-1, X_lstm_test.shape[-1])
            X_test_scaled = self.scaler.transform(X_test_flat)
            X_test_scaled = X_test_scaled.reshape(X_lstm_test.shape)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_callbacks(self, model_dir, patience=20):
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
                patience=12,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(model_dir, f'{self.model_name}_training_log.csv')
            )
        ]
        
        return callbacks_list
    
    def train_dual_input(self, X_cnn_train, X_lstm_train, y_train, 
                        X_cnn_val, X_lstm_val, y_val, 
                        epochs=120, batch_size=32, model_dir='models/hybrid'):
        """
        Train the dual-input hybrid model
        
        Args:
            X_cnn_train, X_lstm_train: Training data for CNN and LSTM branches
            y_train: Training labels (one-hot encoded)
            X_cnn_val, X_lstm_val: Validation data for CNN and LSTM branches
            y_val: Validation labels (one-hot encoded)
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
        
        print(f"Training {self.model_name} model...")
        print(f"Training samples: {X_cnn_train.shape[0]}")
        print(f"Validation samples: {X_cnn_val.shape[0]}")
        print(f"CNN input shape: {X_cnn_train.shape[1:]}")
        print(f"LSTM input shape: {X_lstm_train.shape[1:]}")
        
        # Train the model
        history = self.model.fit(
            [X_cnn_train, X_lstm_train], y_train,
            validation_data=([X_cnn_val, X_lstm_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
        
        # Save training history
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, f'{self.model_name}_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def train_sequential(self, X_train, y_train, X_val, y_val, 
                        epochs=120, batch_size=32, model_dir='models/hybrid'):
        """
        Train the sequential hybrid model (single input)
        
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
            self.build_sequential_hybrid_model()
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks(model_dir)
        
        # Data augmentation for spectrograms
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            zoom_range=0.05,
            fill_mode='nearest'
        )
        
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
            X_test: Test data (single array or list of arrays for dual input)
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
            X: Input data (single array or list of arrays for dual input)
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Normalize LSTM features if scaler is available and dual input
        if isinstance(X, list) and len(X) == 2 and self.scaler is not None:
            X_cnn, X_lstm = X
            X_lstm_flat = X_lstm.reshape(-1, X_lstm.shape[-1])
            X_lstm_scaled = self.scaler.transform(X_lstm_flat)
            X_lstm = X_lstm_scaled.reshape(X_lstm.shape)
            X = [X_cnn, X_lstm]
        
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
            sample_count = len(X) if not isinstance(X, list) else len(X[0])
            for i in range(sample_count):
                prob_dict = {self.class_names[j]: float(y_pred_proba[i][j]) for j in range(self.num_classes)}
                probabilities.append(prob_dict)
            results['class_probabilities'] = probabilities
        
        return results
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.model_name,
            'cnn_input_shape': self.cnn_input_shape,
            'lstm_input_shape': self.lstm_input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'has_scaler': self.scaler is not None,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model
        self.model = keras.models.load_model(filepath)
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_name = metadata.get('model_name', self.model_name)
                self.cnn_input_shape = tuple(metadata.get('cnn_input_shape', self.cnn_input_shape))
                self.lstm_input_shape = tuple(metadata.get('lstm_input_shape', self.lstm_input_shape))
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Hybrid CNN-LSTM Model - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_model_architecture(self, output_path=None):
        """
        Visualize model architecture
        
        Args:
            output_path (str): Path to save the visualization
        """
        if self.model is None:
            self.build_model()
        
        try:
            keras.utils.plot_model(
                self.model,
                to_file=output_path or f'{self.model_name}_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            print(f"Model architecture saved to {output_path or f'{self.model_name}_architecture.png'}")
        except ImportError:
            print("pydot and graphviz are required for model visualization")
            print("Install with: pip install pydot graphviz")