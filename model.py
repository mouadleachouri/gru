"""Main model logic."""

from __future__ import annotations

import tensorflow as tf
from keras import Model
from keras.layers import Concatenate, Dense


class GRUCell(Model):
    """A GRU Cell."""

    def __init__(
        self: GRUCell,
        units: int,
    ) -> None:
        """Initialize a GRUCell instance.

        Parameters
        ----------
        units : int
            Number of units.

        """
        super().__init__()
        self._dense_candidate = Dense(
            name="dense_candidate",
            units=units,
            activation="tanh",
        )
        self._dense_update = Dense(
            name="dense_update",
            units=units,
            activation="sigmoid",
        )

    def call(
        self: GRUCell,
        input_embedding: tf.Tensor,
        prev_hidden_state: tf.Tensor,
    ) -> tf.Tensor:
        """GRU cell forward.

        Parameters
        ----------
        input_embedding: tf.Tensor
            Input embedding.
        prev_hidden_state: tf.Tensor
            Previous hidden state.

        Returns
        -------
        tensorflow.Tensor
            Current hidden state.

        """
        concatenated_inputs = Concatenate(axis=1)([input_embedding, prev_hidden_state])
        hidden_state_candidate = self._dense_candidate(concatenated_inputs)
        update_gate = self._dense_update(concatenated_inputs)
        return (
            update_gate * hidden_state_candidate + (1 - update_gate) * prev_hidden_state
        )


class GRUModel(Model):
    """GRU Model."""

    def __init__(
        self: GRUModel,
        units: int,
        output_size: int,
    ) -> None:
        """Build the GRU model architecture.

        Parameters
        ----------
        output_size : int
            Number of output units.
        units : int
            Number of units.

        """
        super().__init__()
        self._units = units
        self._gru_cell = GRUCell(units=units)
        self._dense_output = Dense(
            name="dense_output",
            units=output_size,
            activation="softmax",
        )

    def call(
        self: GRUModel,
        input_sequence: tf.Tensor,
    ) -> tf.Tensor:
        """GRU model forward.

        Parameters
        ----------
        input_sequence : tf.Tensor
            Input sequence of embeddings.

        Returns
        -------
        tf.Tensor
            A float in the range [0, 1].

        """
        batch_size = tf.shape(input_sequence)[0]
        seq_length = input_sequence.shape[-1]
        hidden_state = tf.zeros(shape=(batch_size, self._units))
        for t in range(seq_length):
            input_embedding = input_sequence[:, :, t]  # ty:ignore[not-subscriptable]
            hidden_state = self._gru_cell(
                input_embedding=input_embedding,
                prev_hidden_state=hidden_state,
            )
        return self._dense_output(hidden_state)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    model = GRUModel(
        output_size=10,
        units=128,
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    rng = np.random.default_rng(seed=42)
    all_inputs = rng.normal(loc=5, scale=5, size=(1000, 100, 10))
    all_outputs = tf.one_hot(rng.uniform(low=0, high=9, size=(1000,)), 10)

    history = model.fit(
        x=all_inputs,
        y=all_outputs,
        validation_split=0.1,
        batch_size=32,
        epochs=100,
    )
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.show()
