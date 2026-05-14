import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
SOURCE_DIR = r"D:\Downloads\Forest_GEC\Forest_GEC\Leaf"
WORK_DIR = "leaf_dataset"
TRAIN_DIR = os.path.join(WORK_DIR, "train")
TEST_DIR = os.path.join(WORK_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SPLIT_RATIO = 0.8  # 80% train, 20% test

CLASS_MAP = {
    "Green leaves": "green",
    "Dry leaves": "dry"
}

# ---------------- CREATE FOLDERS ----------------
for split in ["train", "test"]:
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(WORK_DIR, split, cls), exist_ok=True)

# ---------------- SPLIT DATA ----------------
for src_class, target_class in CLASS_MAP.items():
    src_path = os.path.join(SOURCE_DIR, src_class)
    images = os.listdir(src_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(src_path, img),
            os.path.join(TRAIN_DIR, target_class, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(src_path, img),
            os.path.join(TEST_DIR, target_class, img)
        )

print("✅ Dataset split completed")

# ---------------- DATA GENERATORS ----------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ---------------- CNN MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# ---------------- SAVE MODEL ----------------
model.save("leaf_model.h5")
print("🎉 leaf_model.h5 CREATED SUCCESSFULLY")
