''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
from os.path import join, basename, dirname, exists
import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    

    model.summary()

    return model


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)

    validation_data = MnistGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model = build_model(encoder_path, image_shape=(image_size, image_size, 3), learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )
    # Add t-SNE visualization
    # print('*'*60)
    # print(model.layers)
    # print(model.layers[1].name)
    x, y = validation_data.mnist_handler.get_batch('valid', 100, image_size, color, True)
    print(x.shape)
    print(y.shape)
    model2 = keras.models.Model(inputs=model.input, outputs=model.layers[1].output)
    test_ds = np.concatenate(list(validation_data.take(10).map(lambda x, y : x))) # get five batches of images and convert to numpy array
    print(validation_data.take(10))
    print(test_ds.shape)
    features = model2(test_ds)
    labels = np.argmax(model(test_ds), axis=-1)
    tsne = TSNE(n_components=2).fit_transform(features)

    def scale_to_01_range(x):

        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    benchmark_model(
        encoder_path='models/64x64/encoder.h5',
        epochs=0,
        batch_size=64,
        output_dir='models/64x64',
        lr=1e-3,
        image_size=64,
        color=True
    )
