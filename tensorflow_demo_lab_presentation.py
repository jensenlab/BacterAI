import tensorflow as tf

((x_train, y_train), (x_test, y_test),) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
print("shape", x_train.shape[1])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
# print(f"Training data: {x_train.shape}")
# print(f"Test data: {x_test.shape}")
# print(f"Training sample: {x_train[0]}")
# print(f"Training target sample: {y_train[0]}")

model.compile(
    loss="mse", optimizer="adam", metrics=["mae"],
)
model.fit(x_train, y_train, epochs=100, batch_size=10)

(loss, mae) = model.evaluate(x_test, y_test)

# new_data = # READ IN DATA YOU WANT TO PREDICT
# predictions = model.predict(new_data)
print("Test Set Mean Abs Error: ${:.2f}".format(mae * 1000))
