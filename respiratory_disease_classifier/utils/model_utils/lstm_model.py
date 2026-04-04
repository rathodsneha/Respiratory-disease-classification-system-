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

class LSTMRespiratoryClassifier:
    """LSTM model for respiratory disease classification from sequential audio features"""
    
    def __init__(self, input_shape=(100, 39), num_classes=6, model_name="LSTM_Respiratory"):
        """
        Initialize LSTM model
        
        Args:
            input_shape (tuple): Input shape for sequential features (timesteps, features)
            num_classes (int): Number of disease classes
            model_name (str): Name of the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.scaler = None
        self.class_names = ['Healthy', 'Asthma', 'Pneumonia', 'Bronchitis', 'COPD', 'Pleural_Effusion']
        
    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build LSTM architecture optimized for respiratory sound classification
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First LSTM layer with return sequences
            layers.LSTM(128, return_sequences=True, dropout=dropout_rate, 
                       recurrent_dropout=dropout_rate * 0.5,
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            
            # Second LSTM layer with return sequences
            layers.LSTM(96, return_sequences=True, dropout=dropout_rate,
                       recurrent_dropout=dropout_rate * 0.5,
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            
            # Third LSTM layer with return sequences for attention
            layers.LSTM(64, return_sequences=True, dropout=dropout_rate,
                       recurrent_dropout=dropout_rate * 0.5,
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            
            # Attention mechanism
            layers.Dense(1, activation='tanh'),
            layers.Flatten(),
            layers.Activation('softmax'),
            layers.RepeatVector(64),
            layers.Permute([2, 1]),
            layers.Multiply(),
            layers.Lambda(lambda x: tf.reduce_sum(x, axis=1)),
            
            # Dense layers for classification
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.5),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.3),
            
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
    
    def build_bidirectional_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build Bidirectional LSTM architecture
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate * 0.5,
                           kernel_regularizer=regularizers.l2(0.001))
            ),
            layers.BatchNormalization(),
            
            # Second Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(48, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate * 0.5,
                           kernel_regularizer=regularizers.l2(0.001))
            ),
            layers.BatchNormalization(),
            
            # Third Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate * 0.5,
                           kernel_regularizer=regularizers.l2(0.001))
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu',
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
    
    def prepare_sequential_data(self, features_list, labels_list, sequence_length=100):
        """
        Prepare sequential data from audio features
        
        Args:
            features_list (list): List of feature dictionaries
            labels_list (list): List of labels
            sequence_length (int): Length of sequences to create
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        # Extract relevant features for LSTM
        feature_keys = [
            'mfcc_mean', 'mfcc_std', 'delta_mfcc_mean', 'delta_delta_mfcc_mean',
            'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean',
            'spectral_contrast_mean', 'chroma_mean', 'zero_crossing_rate',
            'energy_mean', 'energy_std', 'wheeze_hf_ratio_mean', 'crackle_energy_variance_mean'
        ]
        
        sequences = []
        sequence_labels = []
        
        for i, (features, label) in enumerate(zip(features_list, labels_list)):
            # Create feature vector
            feature_vector = []
            
            for key in feature_keys:
                if key in features:
                    value = features[key]
                    if isinstance(value, list):
                        feature_vector.extend(value)
                    else:
                        feature_vector.append(value)
                else:
                    # Fill missing features with zeros
                    if 'mfcc' in key:
                        feature_vector.extend([0.0] * 13)  # MFCC has 13 coefficients
                    elif 'spectral_contrast' in key:
                        feature_vector.extend([0.0] * 7)   # Spectral contrast has 7 bands
                    elif 'chroma' in key:
                        feature_vector.extend([0.0] * 12)  # Chroma has 12 bins
                    else:
                        feature_vector.append(0.0)
            
            # Create sequences by sliding window or padding/truncating
            if len(feature_vector) >= sequence_length:
                # If feature vector is long enough, create sliding windows
                for start in range(0, len(feature_vector) - sequence_length + 1, sequence_length // 2):
                    seq = feature_vector[start:start + sequence_length]
                    sequences.append(seq)
                    sequence_labels.append(label)
            else:
                # Pad shorter sequences
                padded_seq = feature_vector + [0.0] * (sequence_length - len(feature_vector))
                sequences.append(padded_seq)
                sequence_labels.append(label)
        
        # Convert to numpy arrays
        X_sequences = np.array(sequences)
        y_sequences = np.array(sequence_labels)
        
        # Reshape for LSTM input (samples, timesteps, features)
        if len(X_sequences.shape) == 2:
            # Calculate number of features per timestep
            total_features = X_sequences.shape[1]
            features_per_timestep = total_features // sequence_length
            
            if features_per_timestep == 0:
                features_per_timestep = 1
                timesteps = total_features
            else:
                timesteps = sequence_length
            
            X_sequences = X_sequences.reshape(-1, timesteps, features_per_timestep)
        
        return X_sequences, y_sequences
    
    def normalize_features(self, X_train, X_val=None, X_test=None):
        """
        Normalize features using StandardScaler
        
        Args:
            X_train (np.array): Training features
            X_val (np.array): Validation features
            X_test (np.array): Test features
            
        Returns:
            tuple: Normalized features
        """
        # Reshape for scaling
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        results = [X_train_scaled]
        
        # Transform validation data
        if X_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            results.append(X_val_scaled)
        
        # Transform test data
        if X_test is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.scaler.transform(X_test_flat)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
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
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(model_dir, f'{self.model_name}_training_log.csv')
            )
        ]
        
        return callbacks_list
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=64, model_dir='models/lstm'):
        """
        Train the LSTM model
        
        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training labels (one-hot encoded)
            X_val (np.array): Validation sequences
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
        
        print(f"Training {self.model_name} model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Input shape: {X_train.shape[1:]}")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
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
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.array): Test sequences
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
            X (np.array): Input sequences
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Normalize features if scaler is available
        if self.scaler is not None:
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(X.shape)
        
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
            'input_shape': self.input_shape,
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('LSTM Model - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_attention_weights(self, X, sample_index=0):
        """
        Extract attention weights for interpretability
        
        Args:
            X (np.array): Input sequences
            sample_index (int): Index of sample to analyze
            
        Returns:
            np.array: Attention weights
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Find attention layer
        attention_layer = None
        for layer in self.model.layers:
            if 'multiply' in layer.name.lower():
                attention_layer = layer
                break
        
        if attention_layer is None:
            raise ValueError("No attention layer found in model")
        
        # Create model to output attention weights
        attention_model = keras.Model(
            inputs=self.model.input,
            outputs=attention_layer.output
        )
        
        # Get attention weights
        attention_weights = attention_model.predict(X[sample_index:sample_index+1])
        
        return attention_weights[0]
    
    def visualize_attention(self, X, sample_index=0, output_path=None):
        """
        Visualize attention weights
        
        Args:
            X (np.array): Input sequences
            sample_index (int): Index of sample to visualize
            output_path (str): Path to save the visualization
        """
        try:
            attention_weights = self.get_attention_weights(X, sample_index)
            
            plt.figure(figsize=(12, 6))
            plt.plot(attention_weights)
            plt.title(f'LSTM Attention Weights - Sample {sample_index}')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Could not visualize attention: {str(e)}")
            print("This model may not have attention mechanism implemented.")