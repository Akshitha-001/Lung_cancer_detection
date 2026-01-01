import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classification_cnn import cnn_classifier

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Model
model = cnn_classifier()
model.summary()

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save model
model.save("lung_cancer_cnn_model.h5")
print("Model Saved Successfully")
