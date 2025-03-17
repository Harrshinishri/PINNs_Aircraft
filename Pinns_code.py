# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# SET SEED FOR REPRODUCIBILITY
np.random.seed(42)
tf.random.set_seed(42)

# LOAD DATASETS
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# CLEAN COLUMN NAMES
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# DEFINE FEATURES & TARGETS
features = ['x', 'y', 'z', 'f']
targets = ['S-Mises', 'S11', 'S22', 'S33', 'S12', 'S13', 'S23']

# NORMALIZE DATA (Using MinMaxScaler)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_features_scaled = feature_scaler.fit_transform(train_df[features])
test_features_scaled = feature_scaler.transform(test_df[features])
train_targets_scaled = target_scaler.fit_transform(train_df[targets])
test_targets_scaled = target_scaler.transform(test_df[targets])

# CONVERT TO TENSORS
train_features_tensor = tf.convert_to_tensor(train_features_scaled, dtype=tf.float32)
test_features_tensor = tf.convert_to_tensor(test_features_scaled, dtype=tf.float32)
train_targets_tensor = tf.convert_to_tensor(train_targets_scaled, dtype=tf.float32)
test_targets_tensor = tf.convert_to_tensor(test_targets_scaled, dtype=tf.float32)

# SAVE NORMALIZED DATASETS (Optional)
train_features_scaled_df = pd.DataFrame(train_features_scaled, columns=features)
train_targets_scaled_df = pd.DataFrame(train_targets_scaled, columns=targets)
test_features_scaled_df = pd.DataFrame(test_features_scaled, columns=features)
test_targets_scaled_df = pd.DataFrame(test_targets_scaled, columns=targets)

train_features_scaled_df.to_csv("train_features_scaled.csv", index=False)
train_targets_scaled_df.to_csv("train_targets_scaled.csv", index=False)
test_features_scaled_df.to_csv("test_features_scaled.csv", index=False)
test_targets_scaled_df.to_csv("test_targets_scaled.csv", index=False)

print(" Data Normalization Completed.")

# DEFINE THE PINNs MODEL
def create_pinn_model(input_shape, output_shape):
    inputs = keras.Input(shape=(input_shape,))

    x = layers.Dense(80, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(80, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(80, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(80, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)

    outputs = layers.Dense(output_shape, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# INITIALIZE MODEL
input_shape = len(features)
output_shape = len(targets)
pinn_model = create_pinn_model(input_shape, output_shape)

# FIXED PHYSICS-INFORMED LOSS FUNCTION
def physics_loss(y_true, y_pred):
    # Compute L_d (Data-based loss)
    L_d = tf.reduce_mean(tf.square(y_pred - y_true))

    # Compute Gradients for Physics Constraints
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_features_tensor)
        y_pred_computed = pinn_model(train_features_tensor)

    dS_dx = tape.gradient(y_pred_computed, train_features_tensor)[:, 0]  # ∂σ/∂x
    dS_dy = tape.gradient(y_pred_computed, train_features_tensor)[:, 1]  # ∂σ/∂y
    dS_dz = tape.gradient(y_pred_computed, train_features_tensor)[:, 2]  # ∂σ/∂z

    del tape  # Free memory

    # Compute L_PDE (Physics loss based on governing equation)
    Re = tf.reduce_mean(tf.square(dS_dx))  # Elasticity residual
    Rc = tf.reduce_mean(tf.square(dS_dy))  # Constraint in y-direction
    Rg = tf.reduce_mean(tf.square(dS_dz))  # Constraint in z-direction

    L_pde = (Re + Rc + Rg) / 3  # Averaging physics residuals

    # Total Loss: Combining L_d and L_pde
    lambda_factor = 0.7  # As per paper
    total_loss = lambda_factor * L_d + (1 - lambda_factor) * L_pde
    return total_loss

# LEARNING RATE DECAY
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)  

# COMPILE MODEL
pinn_model.compile(optimizer=optimizer, loss=physics_loss)

# TRAIN THE MODEL
print(" Training Started...")
history = pinn_model.fit(
    train_features_tensor, train_targets_tensor,
    epochs=50000, batch_size=512, validation_split=0.2, verbose=0,  # 🔄 Suppressing per-epoch output
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)]
)

# PLOT TRAINING LOSS CURVE
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.grid()
plt.show()

# EVALUATE MODEL ON TEST SET
test_loss = pinn_model.evaluate(test_features_tensor, test_targets_tensor, verbose=1)
print(f" Final Test Loss: {test_loss:.6f}")

# SAVE TRAINED MODEL
pinn_model.save("/content/pinns_tf_optimized_model.h5")
print(" Optimized PINNs Model Saved.")

# COMPUTE ERROR METRICS
predictions = pinn_model.predict(test_features_tensor)
predictions_original = target_scaler.inverse_transform(predictions)
actual_original = target_scaler.inverse_transform(test_targets_scaled)

def compute_metrics(pred, actual):
    nrmse = np.sqrt(np.mean((pred - actual) ** 2)) / (np.mean(np.abs(actual)) + 1e-8) * 100
    nmbe = np.mean(pred - actual) / (np.mean(actual) + 1e-8) * 100
    rem = np.mean(np.abs(pred - actual) / (np.max(np.abs(actual)) + 1e-8) * 100)
    return nrmse, nmbe, rem

# Compute and Print NRMSE, NMBE, and REm
nrmse, nmbe, rem = compute_metrics(predictions_original, actual_original)

print(f" NRMSE: {nrmse:.2f}%")
print(f" NMBE: {nmbe:.2f}%")
print(f" REm: {rem:.2f}%")
print(" Error Metric Calculation Completed!")