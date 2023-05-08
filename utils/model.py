import wandb

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


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
    # Start W&B Run
    wandb.init(
        project="deepconvchess",
        config={
            "architecture": model_architecture_class.__name__,
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "loss": "sparse_categorical_crossentropy",
            "metric": "accuracy",
            "batch_size": 32,
            "epochs": 5,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test)
        }
    )

    config = wandb.config

    # Create Model
    resnet_model = model_architecture_class(input_shape=(160, 160, 3), include_top=False)
    resnet_model.trainable = False

    model = Sequential([
        resnet_model, 
        Flatten(), 
        Dense(NUM_CLASSES, activation='softmax', name='output') # Additional output layer for chess pieces
    ], name=config.architecture)

    # Create Optimizer
    if config.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    # Compile Model
    model.compile(optimizer=optimizer, loss=config.loss, metrics=[config.metric])

    # Build Model
    with tf.device('/GPU:0'):
        history = model.fit(X_train, 
                            y_train,
                            batch_size=config.batch_size,
                            epochs=config.epochs,
                            validation_data=(X_val, y_val),
                            callbacks=[EpochLogCallback()]) # Log statistics to W&B

    # Finish W&B Run
    wandb.finish()

    return model