import types

import tensorflow as tf


def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to fit().
    x, y_true = data
    y_pseudo = tf.reshape(tf.argmax(self(x, training=False), axis=1), (-1, 1))

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in compile())
        loss = self.compiled_loss(y_pseudo, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y_true, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}


def pseudo_train_model(model):
    model.train_step = types.MethodType(train_step, model)
    return model