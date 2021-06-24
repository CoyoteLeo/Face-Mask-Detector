import argparse

from keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input as mobilenet_v3_preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

preprocess_input = mobilenet_v3_preprocess_input
model_class = MobileNetV3Large


def main(loaded, fixed, dataset_name, dataset_path):
    # Change the working directory from src to root if needed
    current_full_dir = os.getcwd()
    print("Current working directory: " + current_full_dir)
    if current_full_dir.split("/")[-1] == "src":
        root = current_full_dir[:-4]
        os.chdir(root)
        print("Changed working directory to: " + root)

    # Initialize number of classes and labels
    NUM_CLASS = 3
    class_names = [
        "face_with_mask_incorrect",
        "face_with_mask_correct",
        "face_no_mask",
    ]

    # Initialize the initial learning rate, number of epochs to train for, and batch size
    LEARNING_RATE = 1e-4
    EPOCHS = 50 if dataset_name == "MFN_small" else 20
    BATCH_SIZE = 28
    IMG_SIZE = 224
    checkpoint_filepath = (
        "./checkpoint_"
        + f"{'loaded' if loaded else 'not-loaded'}_"
        + f"{'fixed' if fixed else 'not-fixed'}_"
        + dataset_name
        + "/epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}.h5"
    )
    model_save_path = (
        "./mask_detector_models/mask_detector_"
        + f"{'loaded' if loaded else 'not-loaded'}_"
        + f"{'fixed' if fixed else 'not-fixed'}_"
        + dataset_name
        + ".h5"
    )
    figure_save_path = (
        "./figures/train_plot_"
        + f"{'loaded' if loaded else 'not-loaded'}_"
        + f"{'fixed' if fixed else 'not-fixed'}_"
        + dataset_name
        + ".jpg"
    )

    print("Num of classes: " + str(NUM_CLASS))
    print("Classes: " + str(class_names))
    print("Dataset path: " + dataset_path)
    print("Checkpoint: " + checkpoint_filepath)
    print("Figure save path: " + figure_save_path)

    # Construct the training/validation image generator for data augmentation
    train_data_generator = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        channel_shift_range=10,
        preprocessing_function=preprocess_input,
        fill_mode="nearest",
    )
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,)

    # Set as training data
    train_generator = train_data_generator.flow_from_directory(
        f"{dataset_path}/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=True,
    )

    # Set as validation data
    validation_generator = test_data_generator.flow_from_directory(
        f"{dataset_path}/valid",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )

    # Create classification report
    test_generator = test_data_generator.flow_from_directory(
        f"{dataset_path}/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )

    def test(epoch, _):
        prediction = model.predict_generator(test_generator, verbose=1.0)
        y_pred = np.argmax(prediction, axis=1)
        metric = classification_report(
            test_generator.classes, y_pred, target_names=class_names, output_dict=True
        )
        file_name = (
                "./checkpoint_"
                + f"{'loaded' if loaded else 'not-loaded'}_"
                + f"{'fixed' if fixed else 'not-fixed'}_"
                + dataset_name
                + f"/epoch-{epoch:02d}-val_acc-{metric['weighted avg']['f1-score']:.4f}.txt"
        )
        with open(file_name, "w") as f:
            f.write(
                classification_report(
                    test_generator.classes, y_pred, target_names=class_names
                )
            )

    test_callback = LambdaCallback(
        on_epoch_begin=None,
        on_epoch_end=test,
        on_batch_begin=None,
        on_batch_end=None,
        on_train_begin=None,
        on_train_end=None,
    )

    # Load the pre-trained model and remove the head FC layer
    base_model = model_class(
        weights="imagenet" if loaded else None,
        include_top=False,
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        pooling="max",
    )

    # Construct the head of the model that will be placed on top of the base model
    head_model = base_model.output
    # head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dropout(0.5)(head_model)
    # head_model = Dense(32, activation="relu")(head_model)
    # head_model = Dropout(0.5)(head_model)
    head_model = Dense(NUM_CLASS, activation="softmax")(head_model)

    # Place the head FC model on top of the base model (this will become the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
    if fixed:
        for layer in base_model.layers:
            layer.trainable = False

    # Compile our model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Add early stopping criterion
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.0001,
        patience=3,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # Add model checkpoint
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=False,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="auto",
    )

    # Train the head of the network
    print("[INFO] training head...")
    H = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        callbacks=[early_stopping, checkpoint, test_callback],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples
        // validation_generator.batch_size,
        epochs=EPOCHS,
        workers=12,
        use_multiprocessing=True,
        max_queue_size=32 * 8,
    )

    # Save best model
    model.save(model_save_path)
    # model.load_weights(model_save_path)

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss"
    )
    plt.plot(
        np.arange(0, len(H.history["val_loss"])),
        H.history["val_loss"],
        label="val_loss",
    )
    plt.plot(
        np.arange(0, len(H.history["accuracy"])),
        H.history["accuracy"],
        label="train_acc",
    )
    plt.plot(
        np.arange(0, len(H.history["val_accuracy"])),
        H.history["val_accuracy"],
        label="val_acc",
    )
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(figure_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loaded', action='store_true')
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--dataset_path')
    args = parser.parse_args()
    dataset_name = os.path.basename(args.dataset_path)
    main(args.loaded, args.fixed, dataset_name, args.dataset_path)
