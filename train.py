import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import joblib

class FederatedLearningFraudDetection:
    def __init__(self, train1_path, train2_path, label1_path, label2_path):
        # Check and configure GPU
        self._configure_gpu()
        
        # Load and preprocess data
        self.train1 = self._preprocess_data(pd.read_csv(train1_path))
        self.train2 = self._preprocess_data(pd.read_csv(train2_path))
        
        # Load labels
        self.label1 = pd.read_csv(label1_path).values.ravel().astype(np.int32)
        self.label2 = pd.read_csv(label2_path).values.ravel().astype(np.int32)
        
        # Preprocessing
        self.preprocessor = self._create_preprocessor()
    
    def _configure_gpu(self):
        """Configure GPU settings with error handling"""
        try:
            # Detect available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                try:
                    # Try to use memory growth strategy
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Using {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("No GPU found. Falling back to CPU.")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    def _preprocess_data(self, df):
        """Preprocess raw input data"""
        # Drop or handle unnecessary columns
        columns_to_drop = ['nameOrig', 'nameDest']
        df_processed = df.drop(columns=columns_to_drop, errors='ignore')
        
        return df_processed
    
    def _create_preprocessor(self):
        """Create a preprocessing pipeline"""
        # Identify column types
        numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest']
        categorical_features = ['type']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[ 
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ])
        
        return preprocessor
    
    def apply_smote(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def create_model(self, input_shape):
        """Create Neural Network Model"""
        model = tf.keras.Sequential([ 
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        return model
    
    def federated_learning(self):
        """Perform federated learning"""
        # Preprocess data
        X1_preprocessed = self.preprocessor.fit_transform(self.train1)
        X2_preprocessed = self.preprocessor.transform(self.train2)
        joblib.dump(self.preprocessor, 'preprocessor.pkl')
        
        # Apply SMOTE
        X1_resampled, y1_resampled = self.apply_smote(X1_preprocessed, self.label1)
        X2_resampled, y2_resampled = self.apply_smote(X2_preprocessed, self.label2)
        
        # Split into training and validation sets
        X1_train, X1_val, y1_train, y1_val = train_test_split(
            X1_resampled, y1_resampled, test_size=0.2, random_state=42
        )
        
        X2_train, X2_val, y2_train, y2_val = train_test_split(
            X2_resampled, y2_resampled, test_size=0.2, random_state=42
        )
        
        # Create models for each dataset
        with tf.device('/CPU:0'):  # Fallback to CPU if GPU fails
            # Create and train local models
            model1 = self.create_model(X1_train.shape[1])
            model2 = self.create_model(X2_train.shape[1])
            
            # Early stopping and model checkpointing
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            # Train local models
            model1.fit(X1_train, y1_train, validation_data=(X1_val, y1_val), 
                       epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
            
            model2.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), 
                       epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
            
            # Save model weights
            model1.save_weights("model1.weights.h5")
            model2.save_weights("model2.weights.h5")
            
            # Ensemble model using RandomForest
            rf1 = RandomForestClassifier(n_estimators=400, random_state=42)
            rf2 = RandomForestClassifier(n_estimators=400, random_state=42)
            rf1.fit(X1_train, y1_train)
            rf2.fit(X2_train, y2_train)
            
            # Use VotingClassifier for ensemble
            ensemble_model = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2)], voting='hard')
            ensemble_model.fit(np.concatenate([X1_train, X2_train]), np.concatenate([y1_train, y2_train]))
            
            # Save ensemble model
            joblib.dump(ensemble_model, "ensemble_model.pkl")
            
            return ensemble_model

def main():
    try:
        # Initialize and run Federated Learning
        fl = FederatedLearningFraudDetection(
            'train1.csv', 'train2.csv', 
            'label1.csv', 'label2.csv'
        )
        
        # Run Federated Learning
        ensemble_model = fl.federated_learning()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
