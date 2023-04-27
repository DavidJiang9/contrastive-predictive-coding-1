''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
from os.path import join, basename, dirname, exists
import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    # print('input shape', x_input.shape)
    x1 = encoder(x_input)
    
    # print('encoder shape', x.shape)
    x2 = keras.layers.Dense(units=128, activation='linear')(x1)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.LeakyReLU()(x2)
    x2 = keras.layers.Dense(units=10, activation='softmax')(x2)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x2)
    model2 = keras.models.Model(inputs=x_input, outputs=x1)
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    

    model.summary()
    model2.summary()
    return model, model2


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)

    validation_data = MnistGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model, model2 = build_model(encoder_path, image_shape=(image_size, image_size, 3), learning_rate=lr)

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
    from keras.utils import plot_model 
    plot_model(model, to_file='model.png')
    # design data for plot, 20 points for each of 10 classes
    sets = []
    for i in range(10):
        for j in range(20):
            sets.append(i)
    print(sets)
    x, y = validation_data.mnist_handler.get_batch_by_labels('valid', sets, image_size, color, True)

    # print(x.shape) (100, 64, 64, 3)
    # print(y.shape) (100,)
    # model2 = keras.models.Model(inputs=keras.layers.Input((image_size, image_size, 3)), outputs=model.layers[1].output)

    features = model2(x)
    labels = np.argmax(model(x), axis=-1)
    tsne = TSNE(n_components=2).fit_transform(features)

    def scale_to_01_range(x):

        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    for i in range(tx.shape[0]):
        plt.scatter(tx[i], ty[i],color=plt.cm.Set3(labels[i]))
    for i in range(tx.shape[0]):
        plt.text(tx[i], ty[i], str(labels[i]), color=plt.cm.Set3(labels[i]), 
                fontdict={'weight': 'bold', 'size': 9})

    plt.legend([str(i+1) for i in range(9)])
    plt.savefig('./tsne.png')

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
