import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    (
        (x_train, y_train),
        (x_test, y_test),
    ) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )
    print(y_train)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    k_reg = 0.005
    b_reg = 0.0
    a_reg = 0.01
    do_rate = 0.005

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l1(k_reg),
                bias_regularizer=tf.keras.regularizers.l1(b_reg),
                activity_regularizer=tf.keras.regularizers.l1(a_reg),
                input_shape=(x_train.shape[1],),
            ),
            tf.keras.layers.Dropout(rate=do_rate),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l1(k_reg),
                bias_regularizer=tf.keras.regularizers.l1(b_reg),
                activity_regularizer=tf.keras.regularizers.l1(a_reg),
            ),
            tf.keras.layers.Dropout(rate=do_rate),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l1(k_reg),
                bias_regularizer=tf.keras.regularizers.l1(b_reg),
                activity_regularizer=tf.keras.regularizers.l1(a_reg),
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    print(f"Training data : {x_train.shape}")
    print(f"Test data : {x_test.shape}")
    print(f"Training sample : {x_train[0]}")
    print(f"Training target sample : {y_train[0]}")

    model.compile(
        loss="mse", optimizer="adam", metrics=["mae"],
    )
    model.fit(x_train, y_train, epochs=100, batch_size=10)
    (loss, mae) = model.evaluate(x_test, y_test)

    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

    k_num = 0
    k_den = 0
    b_num = 0
    b_den = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            continue
        k, b = layer.get_weights()
        k_num += k[np.absolute(k) < 1e-3].size
        k_den += k.size
        b_num += b[np.absolute(b) < 1e-3].size
        b_den += b.size

    k_sparcity = k_num / k_den
    b_sparcity = b_num / b_den
    print("SPARCITY", k_sparcity, b_sparcity)
