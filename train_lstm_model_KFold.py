import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tf_keras as keras
from tf_keras.utils import timeseries_dataset_from_array
import pandas as pd
from sklearn.model_selection import KFold

# Download Data
fname = os.path.join("data/combined_data", "target_combined_data.csv")
with open(fname) as f:
    data = f.read()

def run_experiment(TARGET, SEQUENCE_LENGTH, DELAY, BATCH_SIZE, DROPOUT_RATE, LAYER_SIZE, EPOCH, ACTIVATION, LEARNING_RATE, PATIENCE, display_result, X, y, fold_idx, total_samples):
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

    # Standardize Data
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

    # Adjust total_samples for delay
    effective_samples = len(raw_data_std) - DELAY

    # Create Train/Validate/Test dataset using provided X and y
    sampling_rate = 1
    sequence_length = SEQUENCE_LENGTH
    delay = DELAY
    batch_size = BATCH_SIZE

    # Calculate fold boundaries based on effective samples
    fold_size = effective_samples // n_splits
    start_idx = 0 if fold_idx == 0 else fold_size * fold_idx
    end_idx = fold_size * (fold_idx + 1) if fold_idx < n_splits - 1 else effective_samples

    train_dataset = timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=end_idx if fold_idx == 0 else start_idx,
    )

    val_dataset = timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=start_idx,
        end_index=end_idx,
    )

    test_dataset = timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=end_idx if fold_idx == n_splits - 1 else None,
    )

    all_dataset = timeseries_dataset_from_array(
        X[:-delay],
        targets=y[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=0
    )

    ## Model #4 : LSTM(Long Short-Term Memory) RNN Model w/ dropout
    inputs = keras.Input(shape=(sequence_length, raw_data_std.shape[-1]))
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, return_sequences=True, recurrent_dropout=DROPOUT_RATE)(inputs)
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, return_sequences=True, recurrent_dropout=DROPOUT_RATE)(x)
    x = keras.layers.LSTM(LAYER_SIZE, activation=ACTIVATION, recurrent_dropout=DROPOUT_RATE)(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(f"models/tastemap_lstm_dropout_fold_{fold_idx}.h5", save_best_only=True),
        keras.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * (0.5 ** (epoch // 500))),
    ]
    if PATIENCE > 0:
        callbacks.append(keras.callbacks.EarlyStopping(patience=PATIENCE))

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset, callbacks=callbacks, verbose=0)

    best_epoch = np.argmin(history.history['val_mae']) + 1
    best_val_mae = min(history.history['val_mae'])

    model = keras.models.load_model(f"models/tastemap_lstm_dropout_fold_{fold_idx}.h5")
    eval = model.evaluate(test_dataset, verbose=0)[1]
    actual_mae = eval * std[TARGET] + mean[TARGET]

    return eval, actual_mae, best_epoch, best_val_mae

def run_kfold_cross_validation(n_splits=1):  # Reduced to 1 due to limited data
    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:-1]

    target_value = np.zeros((len(lines),))
    raw_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        target_value[i] = values[TARGET]
        raw_data[i, :] = values[:]

    # Standardize Data
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

    # Reshape for LSTM
    X = raw_data_std
    y = target_value_std
    total_samples = len(X)

    if total_samples < n_splits:
        print(f"Warning: Total samples ({total_samples}) less than n_splits ({n_splits}). Reducing n_splits to {total_samples}.")
        n_splits = total_samples

    # K-Fold Cross Validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training fold {fold_idx + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        eval_score, actual_mae, best_epoch, best_val_mae = run_experiment(
            TARGET, SEQUENCE_LENGTH, DELAY, BATCH_SIZE, DROPOUT_RATE, LAYER_SIZE,
            EPOCH, ACTIVATION, LEARNING_RATE, PATIENCE,
            display_result, X_train, y_train, fold_idx, total_samples
        )
        fold_scores.append((eval_score, actual_mae, best_epoch, best_val_mae))
        print(f"Fold {fold_idx + 1} - Test MAE: {eval_score:.4f}, Actual MAE: {actual_mae:.2f}, Best Epoch: {best_epoch}, Best Val MAE: {best_val_mae:.4f}")

    # Aggregate results
    mean_eval = np.mean([score[0] for score in fold_scores])
    mean_actual_mae = np.mean([score[1] for score in fold_scores])
    mean_best_epoch = np.mean([score[2] for score in fold_scores])
    mean_best_val_mae = np.mean([score[3] for score in fold_scores])

    # Save results
    results = {
        "Run_Date": [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
        "K_Folds": [n_splits],
        "Mean_Test_MAE": [mean_eval],
        "Mean_Actual_Test_MAE": [mean_actual_mae],
        "Mean_Best_Epoch": [mean_best_epoch],
        "Mean_Best_Val_MAE": [mean_best_val_mae],
    }
    df_results = pd.DataFrame(results)
    excel_file = "models/model_results_kfold.xlsx"
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        df_results = pd.concat([existing_df, df_results], ignore_index=True)
    df_results.to_excel(excel_file, index=False)

    if display_result:
        # Visualize average performance
        plt.figure()
        plt.bar(["Mean Test MAE", "Mean Actual MAE"], [mean_eval, mean_actual_mae])
        plt.title("K-Fold Cross Validation Results")
        plt.ylabel("MAE")
        plt.savefig("models/kfold_results_plot.png")
        plt.show()

# HYPERPARAMETERS
TARGET = 50
SEQUENCE_LENGTH = 6
DELAY = SEQUENCE_LENGTH
BATCH_SIZE = 32
DROPOUT_RATE = 0.25
LAYER_SIZE = 64
EPOCH = 2000
ACTIVATION = 'relu'
LEARNING_RATE = 0.001
PATIENCE = 0

display_result = True
n_splits = 2  # Reduced to 1 due to limited data (2 months)

# Run K-Fold Cross Validation
run_kfold_cross_validation(n_splits)