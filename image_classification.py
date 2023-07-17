# -*- coding: utf-8 -*-
"""

@author: manon-col
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import utils


class Model:
    
    def __init__(self, model_path, image_size, batch_size, num_epochs):
        
        self._model_path = model_path
        self._labelled = model_path+'/labelled'
        self._unlabelled = model_path+'/unlabelled'
        
        self._image_size = image_size
        self._input_shape = (self._image_size[0], self._image_size[1], 3)

        self._batch_size = int(batch_size)
        self._num_epochs = int(num_epochs)
    
        self._AUTOTUNE = tf.data.AUTOTUNE
        self._shuffle_buffer = 5000
        
        self._temperature = 0.1
        self._queue_size = 10000
    
        self._contrastive_augmenter = {
            "brightness": 0.5,
            "name": "contrastive_augmenter",
            "scale": (0.2, 1.0),
        }
        
        self._classification_augmenter = {
            "brightness": 0.2,
            "name": "classification_augmenter",
            "scale": (0.5, 1.0),
        }
        
        self._width = 128
    
    def prepare_dataset(self):
       
        labelled_batch_size = self._batch_size // 2
        unlabelled_batch_size = self._batch_size // 2
        
        self._labelled_train_ds = (utils.image_dataset_from_directory(
            self._labelled,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self._image_size,
            )
            .unbatch()
            .batch(batch_size=labelled_batch_size, drop_remainder=True)
            .shuffle(buffer_size=self._shuffle_buffer)
        )
        
        self._val_ds = (utils.image_dataset_from_directory(
            self._labelled,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self._image_size,
            )
            .unbatch()
            .batch(batch_size=self._batch_size, drop_remainder=True)
            .prefetch(buffer_size=self._AUTOTUNE)
        )
        
        self._unlabelled_train_ds = (utils.image_dataset_from_directory(
            self._unlabelled,
            seed=42,
            image_size=self._image_size,
            )
            .unbatch()
            .batch(batch_size=unlabelled_batch_size, drop_remainder=True)
            .shuffle(buffer_size=self._shuffle_buffer)
        )
        
        self._train_ds = tf.data.Dataset.zip(
            (self._unlabelled_train_ds, self._labelled_train_ds)
        ).prefetch(buffer_size=self._AUTOTUNE)
    
    def pretraining(self):
        
        print("Building NNCLR model.")
        
        self._model = NNCLR(temperature=self._temperature,
                            queue_size=self._queue_size,
                            width=self._width,
                            input_shape=self._input_shape,
                            contrastive_augmenter=self._contrastive_augmenter,
                            classification_augmenter=self._classification_augmenter)
        
        self._model.compile(
            contrastive_optimizer=keras.optimizers.Adam(),
            probe_optimizer=keras.optimizers.Adam(),
        )
        
        pretrain_history = self._model.fit(
            self._train_ds, epochs=self._num_epochs, validation_data=self._val_ds,
            callbacks= [
                keras.callbacks.ModelCheckpoint(
                    self._model_path,
                    save_weights_only=True,
                    save_best_only=True,
                    monitor="val_p_loss"),
                tf.keras.callbacks.EarlyStopping(monitor="val_p_loss", patience=5)]
        )
        
        self.plot_history(pretrain_history, "Pretraining History")
    
    def finetuning(self, save_path):
        
        finetuning_model = keras.Sequential(
            [
                layers.Input(shape=self._input_shape),
                augmenter(**self._classification_augmenter,
                          input_shape=self._input_shape),
                self._model.encoder,
                layers.Dense(10),
            ],
            name="finetuning_model",
        )
        
        finetuning_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

        finetuning_history = finetuning_model.fit(
            self._labelled_train_ds, epochs=self._num_epochs,
            validation_data=self._val_ds,
            callbacks= [
                keras.callbacks.ModelCheckpoint(
                    self._model_path,
                    save_weights_only=True,
                    save_best_only=True,
                    monitor="val_loss"),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)] 
        )
        
        self.plot_history(finetuning_history, "Finetuning History")
        
        finetuning_model.save(
            filepath=save_path,
            overwrite=True,
            save_format=None,
            options=None,
            include_optimizer=True,
            signatures=None
        )
    
    def plot_history(self, history, title):
        
        plt.figure(figsize=(12, 4))
        
        if "p_loss" in history.history and "val_p_loss" in history.history:
        
            plt.subplot(1, 2, 1)
            plt.plot(history.history['p_loss'], label='Probe Loss')
            plt.plot(history.history['val_p_loss'], label='Validation Probe Loss')
            plt.title('Probe Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
        
        if "p_acc" in history.history and "val_p_acc" in history.history:
        
            plt.subplot(1, 2, 2)
            plt.plot(history.history['p_acc'], label='Probe Accuracy')
            plt.plot(history.history['val_p_acc'], label='Validation Probe Accuracy')
            plt.title('Probe Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
        
        if "loss" in history.history and "val_loss" in history.history:
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
        if "acc" in history.history and "val_acc" in history.history:
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['acc'], label='Accuracy')
            plt.plot(history.history['val_acc'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.suptitle(title)
        plt.show()
    
    def load(self, filepath):
        
        self._model = tf.keras.models.load_model(filepath, compile=False)
        
        for layer in self._model.layers:
            if not isinstance(layer, layers.Dense):
                layer.trainable = False

        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
            )
        
        self._model.summary()
    
    def _build_pm(self):
        
        self._probability_model = tf.keras.Sequential([self._model,
                                                       layers.Softmax()])
    
    def prediction(self, images, threshold=None):
        """
        Realise image predictions in batch. Return class 0 images (deadwood)
        and class 1 images with a confidence score < threshold.

        Parameters
        ----------
        images : list
            List of images to predict.
        threshold : integer, optional
            Minimum score of confidence tolerated for inclusion in the "other"
            class (class 1). If score < threshold, it is The default is None.

        Returns
        -------
        predictions : list
            List of the input images classified as deadwood.

        """
        # Initialise array of all images
        array = None
        
        if not hasattr(self, '_probability_model'):
            self._build_pm()
        
        for image in images:
            
            img = utils.load_img(image)
            img_array = utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            
            if array is None: array = img_array
            
            else: array = np.concatenate((array, img_array), axis=0)
        
        # Initialise list of deadwood images
        deadwood_images = []
        
        predictions = self._probability_model.predict(array,
                                                      batch_size=len(array),
                                                      verbose=0)   
        
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        predicted_scores = tf.reduce_max(predictions, axis=1).numpy()
        
        for i in range(len(images)):
            if predicted_classes[i] == 0 or (threshold is not None and \
                                             predicted_scores[i] < threshold):
                deadwood_images.append(images[i])
    
        return deadwood_images
        

class RandomResizedCrop(layers.Layer):
    
    def __init__(self, scale, ratio):
        super().__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
        random_ratios = tf.exp(
            tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
        )

        new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
        height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
        width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images, bounding_boxes, tf.range(batch_size), (height, width)
        )
        return images


class RandomBrightness(layers.Layer):
    
    def __init__(self, brightness):
        super().__init__()
        self.brightness = brightness

    def blend(self, images_1, images_2, ratios):
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def call(self, images):
        
        images = self.random_brightness(images)
        return images


def augmenter(brightness, name, scale, input_shape):
    
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=scale, ratio=(3 / 4, 4 / 3)),
            RandomBrightness(brightness=brightness),
        ],
        name=name,
    )


def encoder(input_shape, width):
    
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )

    
class NNCLR(keras.Model):
    
    def __init__(
            
        self, temperature, queue_size, width, input_shape,
        contrastive_augmenter, classification_augmenter
    ):
        super().__init__()
        self._temperature = temperature
        self._queue_size = queue_size
        self._width = width
        self._input_shape = input_shape
        
        self._probe_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self._correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self._contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self._probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self._contrastive_augmenter = augmenter(**contrastive_augmenter,
                                                input_shape=self._input_shape)
        self._classification_augmenter = augmenter(**classification_augmenter,
                                                   input_shape=self._input_shape)
        self.encoder = encoder(input_shape=self._input_shape, width=self._width)
        self._projection_head = keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )
        self._linear_probe = keras.Sequential(
            [layers.Input(shape=(self._width,)), layers.Dense(10)], name="linear_probe"
        )

        feature_dimensions = self.encoder.output_shape[1]
        self._feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(self._queue_size, feature_dimensions)),
                axis=1
            ),
            trainable=False,
        )
        
    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        
        super().compile(**kwargs)
        self._contrastive_optimizer = contrastive_optimizer
        self._probe_optimizer = probe_optimizer
    
    def nearest_neighbour(self, projections):
        
        support_similarities = tf.matmul(
            projections, self._feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self._feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self._contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.cast(tf.shape(features_1)[0], tf.float32)

        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self._correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self._temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self._temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self._temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self._temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self._feature_queue.assign(
            tf.concat([projections_1, self._feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
            
        (unlabelled_images, _), (labelled_images, labels) = data
        images = tf.concat((unlabelled_images, labelled_images), axis=0)
        augmented_images_1 = self._contrastive_augmenter(images)
        augmented_images_2 = self._contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self._projection_head(features_1)
            projections_2 = self._projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self._projection_head.trainable_weights,
        )
        self._contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self._projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self._classification_augmenter(labelled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self._linear_probe(features)
            probe_loss = self._probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self._linear_probe.trainable_weights)
        self._probe_optimizer.apply_gradients(
            zip(gradients, self._linear_probe.trainable_weights)
        )
        self._probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self._contrastive_accuracy.result(),
            "r_acc": self._correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self._probe_accuracy.result(),
        }

    def test_step(self, data):
        
        labelled_images, labels = data

        preprocessed_images = self._classification_augmenter(
            labelled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self._linear_probe(features, training=False)
        probe_loss = self._probe_loss(labels, class_logits)

        self._probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self._probe_accuracy.result()}