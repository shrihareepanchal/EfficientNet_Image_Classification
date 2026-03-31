import os
import matplotlib.pyplot as plt

# Keras utilities for data loading and training
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom model builder (EfficientNet-based)
from model import build_model


# 1. Create output directories for saving results
os.makedirs('outputs/plots', exist_ok=True)


# 2. Define image and training parameters
IMG = 224                 # Standard input size for EfficientNet
BATCH = 8                 # Smaller batch size (better for small dataset)


# 3. Data Preprocessing + Augmentation
# Why augmentation?
# -> Helps increase dataset diversity
# -> Prevents overfitting on small dataset
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values (0–1)
    validation_split=0.2,        # 80% train / 20% validation split
    rotation_range=25,           # Random rotation
    zoom_range=0.3,              # Random zoom
    horizontal_flip=True,        # Flip images horizontally
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1       # Vertical shift
)


# 4. Load dataset from directory

train = datagen.flow_from_directory(
    'data_grouped',
    target_size=(IMG, IMG),
    batch_size=BATCH,
    subset='training'            # Training split
)

val = datagen.flow_from_directory(
    'data_grouped',
    target_size=(IMG, IMG),
    batch_size=BATCH,
    subset='validation'          # Validation split
)


# 5. Build model (EfficientNet + custom head)
model = build_model(train.num_classes)


# 6. Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-3),   # Initial learning rate
    loss='categorical_crossentropy',      # Multi-class classification loss
    metrics=['accuracy']
)


# 7. Callbacks for better training control

# Stops training early if validation loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Reduces learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)


# 8. Train the model
history = model.fit(
    train,
    validation_data=val,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)


# 9. Save trained model
# Using modern Keras format (.keras)
model.save('outputs/model.keras')


# 10. Plot Accuracy Curve
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('outputs/plots/accuracy.png')


# 11. Plot Loss Curve
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('outputs/plots/loss.png')