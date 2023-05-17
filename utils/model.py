import wandb

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from sklearn.metrics import precision_score, recall_score, f1_score


PIECE_LABELS = ['_', 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
NUM_CLASSES = len(PIECE_LABELS)


class EpochLogCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        train_loss = logs.get('loss')
        
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        wandb.log(
            {
                'accuracy': train_acc, 
                'loss': train_loss,
                'val_accuracy': val_acc,
                'val_loss': val_loss
            }
        )

def test_model(model_architecture_class, X_train, y_train, X_val, y_val, X_test, y_test):
    config = {
        "architecture": model_architecture_class.__name__,
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "loss": "sparse_categorical_crossentropy",
        "batch_size": 32,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test)
    }
    
    # Start W&B Run
    with wandb.init(project="deepconvchess", config=config):
        # Create Model
        base_model = model_architecture_class(input_shape=(160, 160, 3), include_top=False)
        base_model.trainable = False

        model = Sequential([
            base_model, 
            Flatten(), 
            Dense(NUM_CLASSES, activation='softmax', name='output') # Additional output layer for chess pieces
        ], name=wandb.config.architecture)

        # Create Optimizer
        if wandb.config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)

        # Compile Model
        model.compile(optimizer=optimizer, loss=wandb.config.loss, metrics=['accuracy'])

        # Early stopping callback
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        # Build Model
        with tf.device('/GPU:0'):
            history = model.fit(X_train, 
                                y_train,
                                batch_size=wandb.config.batch_size,
                                epochs=10,
                                validation_data=(X_val, y_val),
                                callbacks=[
                                    EpochLogCallback(), # Log statistics to W&B
                                    early_stop_callback,
                            ])

        
        # Log test accuracy and loss
        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Log final statistics
        wandb.log(
            {
                'epochs': len(history.epoch),
                'test_accuracy': test_accuracy, 
                'test_loss': test_loss,
                'precision': precision_score(y_test, y_pred_classes, average='weighted'),
                'recall': recall_score(y_test, y_pred_classes, average='weighted'),
                'f1': f1_score(y_test, y_pred_classes, average='weighted'),
                'total_params': model.count_params()
            }
        )

    return model
