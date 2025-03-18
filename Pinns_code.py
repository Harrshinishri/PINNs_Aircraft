# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ENABLE GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')  
        print("TensorFlow is using GPU!")
    except RuntimeError as e:
        print(e)

# SET SEED FOR REPRODUCIBILITY
np.random.seed(42)
tf.random.set_seed(42)

# LOAD DATASETS
train_df = pd.read_csv("/media/ubuntu/CLEM/my_env/Pinns_aircraft/train.csv")
test_df = pd.read_csv("/media/ubuntu/CLEM/my_env/Pinns_aircraft/test.csv")

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

# DEFINE THE PINNs MODEL (5 Hidden Layers, 60 Neurons Each)
def create_pinn_model(input_shape, output_shape):
    inputs = keras.Input(shape=(input_shape,))
   
    x = layers.Dense(60, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    for _ in range(4):  # 5 hidden layers in total
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(60, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
   
    outputs = layers.Dense(output_shape, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# INITIALIZE MODEL
input_shape = len(features)
output_shape = len(targets)
pinn_model = create_pinn_model(input_shape, output_shape)

# PHYSICS-INFORMED LOSS FUNCTION (As per Paper)
def physics_loss(y_true, y_pred):
    # Data-Based Loss (L_d)
    L_d = tf.reduce_mean(tf.square(y_pred - y_true))

    # Compute Gradients for Physics Constraints
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_features_tensor)
        y_pred_computed = pinn_model(train_features_tensor)

    dS_dx = tape.gradient(y_pred_computed, train_features_tensor)[:, 0]
    dS_dy = tape.gradient(y_pred_computed, train_features_tensor)[:, 1]
    dS_dz = tape.gradient(y_pred_computed, train_features_tensor)[:, 2]

    del tape  # Free memory

    # Governing Equation Residuals (L_pde)
    residual_e = tf.reduce_mean(tf.square(dS_dx))
    residual_c = tf.reduce_mean(tf.square(dS_dy))
    residual_g = tf.reduce_mean(tf.square(dS_dz))

    L_pde = (residual_e + residual_c + residual_g) / 3  # Averaging PDE residuals

    # External Force Constraint
    external_forces = feature_scaler.transform(train_df[features])[:, -1].reshape(-1, 1)  
    external_forces = tf.convert_to_tensor(external_forces, dtype=tf.float32)
    external_forces = tf.tile(external_forces, [1, 7])

    force_constraint = tf.reduce_mean(tf.square(y_pred_computed - external_forces))

    # Final Loss Function (λ = 0.7 from Paper)
    lambda_d = 0.7
    total_loss = lambda_d * L_d + (1 - lambda_d) * (L_pde + 0.05 * force_constraint)
    return total_loss

# LEARNING RATE DECAY
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# COMPILE MODEL
pinn_model.compile(optimizer=optimizer, loss=physics_loss)

# CUSTOM CALLBACK FOR LOGGING EVERY 500 EPOCHS
class CustomEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 500 == 0:  # Print every 500 epochs
            print(f"Epoch {epoch}: Training Loss = {logs['loss']:.6f}, Validation Loss = {logs['val_loss']:.6f}")

# TRAIN THE MODEL (50,000 Epochs)
print("Training Started...")
history = pinn_model.fit(
    train_features_tensor, train_targets_tensor,
    epochs=50000, batch_size=512, validation_split=0.2, verbose=0,
    callbacks=[CustomEpochLogger(), keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)]
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
plt.savefig("/media/ubuntu/CLEM/my_env/Pinns_aircraft/training_loss_plot.png")  # Save plot as file

# EVALUATE MODEL ON TEST SET
test_loss = pinn_model.evaluate(test_features_tensor, test_targets_tensor, verbose=1)
print(f"Final Test Loss: {test_loss:.6f}")

# SAVE TRAINED MODEL IN KERAS FORMAT
pinn_model.save("/media/ubuntu/CLEM/my_env/Pinns_aircraft/pinns_tf_optimized_model.keras")
print("Optimized PINNs Model Saved in Keras Format.")

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

print(f"NRMSE: {nrmse:.2f}%")
print(f"NMBE: {nmbe:.2f}%")
print(f"REm: {rem:.2f}%")
print("Error Metric Calculation Completed!")
