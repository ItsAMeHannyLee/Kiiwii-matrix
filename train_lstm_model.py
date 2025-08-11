import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tf_keras as keras
from tf_keras.utils import timeseries_dataset_from_array
from keras import regularizers
import pandas as pd

# Download Data
fname = os.path.join("data/combined_data", "target_combined_data.csv")
with open(fname) as f:
    data = f.read()

def run_experiment(TARGET, SEQUENCE_LENGTH, DELAY, BATCH_SIZE, DROPOUT_RATE, LAYER_SIZE, REGULIZER_L1, REGULIZER_L2, EPOCH, ACTIVATION, LEARNING_RATE, PATIENCE, display_result):
    # Data Parsing
    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:-1]

    target_value = np.zeros((len(lines),))
    raw_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        target_value[i] = values[TARGET]
        raw_data[i, :] = values[:]

    # Extract year_month for indexing
    year_months = [line.split(",")[0] for line in lines[1:-1]]

    # Calculate number of Train[50%]/Validate[25%]/Test[25%] samples
    num_train_samples = int(0.5 * len(raw_data))
    num_val_samples = int(0.25 * len(raw_data))
    num_test_samples = len(raw_data) - num_train_samples - num_val_samples

    # # Standardize Data ignoring values with 3.141592
    raw_data_std = raw_data.copy()
    mean = raw_data_std.mean(axis=0)
    raw_data_std -= mean
    std = raw_data_std.std(axis=0)
    raw_data_std /= np.maximum(std, 1e-7)

    target_value_std = target_value.copy()
    mean_target = target_value.mean()
    std_target = target_value.std()
    target_value_std -= mean_target
    target_value_std /= np.maximum(std_target, 1e-7)

    # # Standardize Data ignoring values with 3.141592
    # raw_data_std = raw_data.copy()
    # mean = raw_data_std[:num_train_samples].mean(axis=0)
    # raw_data_std -= mean
    # std = raw_data_std[:num_train_samples].std(axis=0)
    # raw_data_std /= np.maximum(std, 1e-7)

    # target_value_std = target_value.copy()
    # mean_target = target_value[:num_train_samples].mean()
    # std_target = target_value[:num_train_samples].std()
    # target_value_std -= mean_target
    # target_value_std /= np.maximum(std_target, 1e-7)

    # Create Train/Validate/Test dataset
    sampling_rate = 1
    sequence_length = SEQUENCE_LENGTH
    delay = DELAY  # Predict next month
    batch_size = BATCH_SIZE

    train_dataset = timeseries_dataset_from_array(
        raw_data_std[:-delay],
        targets=target_value_std[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples,
    )

    val_dataset = timeseries_dataset_from_array(
        raw_data_std[:-delay],
        targets=target_value_std[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples,
    )

    test_dataset = timeseries_dataset_from_array(
        raw_data_std[:-delay],
        targets=target_value_std[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=num_train_samples + num_val_samples,
    )

    all_dataset = timeseries_dataset_from_array(
        raw_data_std[:-delay],
        targets=target_value_std[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=0
    )

    ## Model #4 : LSTM(Long Short-Term Memory) RNN Model w/ dropout
    # Define Model
    inputs = keras.Input(shape=(sequence_length, raw_data_std.shape[-1]))
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, return_sequences=True, recurrent_dropout=DROPOUT_RATE, kernel_regularizer=regularizers.l1_l2(l1=REGULIZER_L1, l2=REGULIZER_L2))(inputs)
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, return_sequences=True, recurrent_dropout=DROPOUT_RATE, kernel_regularizer=regularizers.l1_l2(l1=REGULIZER_L1, l2=REGULIZER_L2))(x)
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, recurrent_dropout=DROPOUT_RATE, kernel_regularizer=regularizers.l1_l2(l1=REGULIZER_L1, l2=REGULIZER_L2))(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    # Train Model
    callbacks = [
        keras.callbacks.ModelCheckpoint("models/tastemap_lstm_dropout.h5", save_best_only=True),
        # keras.callbacks.TensorBoard(log_dir="logs/tastemap_lstm_dropout"),
        keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * (0.5 ** (epoch // 500))),  # 학습률 스케줄러
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6), # Reduce learning rate on plateau
    ]
    if PATIENCE > 0:
        callbacks.append(keras.callbacks.EarlyStopping(patience=PATIENCE))
    
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset, callbacks=callbacks)

    # Find the best epoch
    best_epoch = np.argmin(history.history['val_mae']) + 1  # +1 because epoch starts from 1
    best_val_mae = min(history.history['val_mae'])

    # Visualize Model
    model = keras.models.load_model("models/tastemap_lstm_dropout.h5")
    eval = model.evaluate(test_dataset)[1]
    # print(f"Test MAE: {eval:.2f}")

    actual_mae = eval * std[TARGET] + mean[TARGET]
    # print(f"Actual Test MAE: {actual_mae:.2f} tons")

    # Save results to Excel
    results = {
        "Run_Date": [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
        "Target": [TARGET],
        "Sequence_Length": [SEQUENCE_LENGTH],
        "Delay": [DELAY],
        "Batch_Size": [BATCH_SIZE],
        "Dropout_Rate": [DROPOUT_RATE],
        "Layer_Size": [LAYER_SIZE],
        "Regularizer_L1": [REGULIZER_L1],
        "Regularizer_L2": [REGULIZER_L2],
        "Epochs": [EPOCH],
        "Activation": [ACTIVATION],
        "Test_MAE": [eval],
        "Actual_Test_MAE": [actual_mae],
        "Best_Epoch": [best_epoch],
        "Best_Val_MAE": [best_val_mae],
        "Learning_Rate": [LEARNING_RATE],
        "Patience": [PATIENCE],
    }
    df = pd.DataFrame(results)
    excel_file = "models/model_results.xlsx"
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_excel(excel_file, index=False)

    if display_result:
        loss = history.history["mae"]
        val_loss = history.history["val_mae"]
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training MAE")
        plt.plot(epochs, val_loss, "b", label="Validation MAE")
        plt.title("Training and validation MAE")
        plt.legend()
        plt.savefig("models/error_plot.png")
        plt.show()

        # TESTING
        actual_values = target_value
        pred_values = model.predict(all_dataset, batch_size=32)

        # Un-standardize predictions
        predicted_values = pred_values[:, 0] * std_target + mean_target

        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(actual_values) + 1), actual_values, "go-", label="Actual Values", linewidth=2)
        plt.plot(range(1 + delay, len(predicted_values) + 1 + delay), predicted_values, "ro-", label="Predicted Values", linewidth=2)
        plt.title("Actual vs Predicted Export Values (China, Hong Kong SAR)")
        plt.xlabel("Time Steps (Months)")
        plt.ylabel("Export Value (Dollars)")
        plt.legend()
        plt.grid(True)
        plt.savefig("models/prediction_plot.png")
        plt.show()

# HYPERPARAMETERS
TARGET = 50
SEQUENCE_LENGTH = 6
DELAY = SEQUENCE_LENGTH
BATCH_SIZE = 32
DROPOUT_RATE = 0.25
LAYER_SIZE = 64
REGULIZER_L1 = 0.001
REGULIZER_L2 = 0.002
EPOCH = 2000
ACTIVATION = 'relu'
LEARNING_RATE = 0.001
PATIENCE = 0

repeat = 1
display_result = True

for i in range(1, repeat+1):
    # HYPERPARAMETERS
    run_experiment(TARGET, SEQUENCE_LENGTH, DELAY, BATCH_SIZE, DROPOUT_RATE, LAYER_SIZE, REGULIZER_L1, REGULIZER_L2, EPOCH, ACTIVATION, LEARNING_RATE, PATIENCE, display_result)