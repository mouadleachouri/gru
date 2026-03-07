"""Main model logic."""

from __future__ import annotations

import keras
import numpy as np
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
            units=1,
            activation="sigmoid",
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
    # Train/Test data
    num_words, skip_top, maxlen, start_char, oov_char = 1000, 100, 100, 1, 2
    start_from = skip_top + 1
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=num_words,
        skip_top=skip_top,
        maxlen=maxlen,
        start_char=start_char,
        oov_char=oov_char,
    )
    for i in range(len(x_train)):
        x_train[i] = tf.pad(
            x_train[i],
            tf.constant([[0, maxlen - len(x_train[i])]]),
        )
    for i in range(len(x_test)):
        x_test[i] = tf.pad(
            x_test[i],
            tf.constant([[0, maxlen - len(x_test[i])]]),
        )
    x_train, x_test = (
        np.array([np.array(row) for row in x_train]),
        np.array([np.array(row) for row in x_test]),
    )

    # Word/Index mapping
    word_to_index = keras.datasets.imdb.get_word_index()
    index_to_word = {start_from + value: key for key, value in word_to_index.items()}
    index_to_word[1] = "<SOS>"
    index_to_word[2] = "<OOV>"

    # Create TF Datasets
    def _one_hot_map(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = tf.one_hot(x, depth=num_words)
        return x, y

    def _create_ds(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_tensor_slices((x, y))
            .map(_one_hot_map)
            .batch(32)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    ds_train = _create_ds(x_train, y_train)
    ds_test = _create_ds(x_test, y_test)

    # Train the model
    model = GRUModel(units=128)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(ds_train, epochs=100, validation_data=ds_test)
