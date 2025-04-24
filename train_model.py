from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt


def create_lstm(unit=15):

    model = Sequential()
    model.add(Bidirectional(LSTM(unit, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(unit, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Flatten())
    model.add(Dense(1))

    return model


def train_model(model, X_train, Y_train, X_valid, Y_valid, epochs=200, batch_size=500):

    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mse', optimizer=opt)
    EarlyStop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
    history = model.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], verbose=1,
                        epochs=epochs, batch_size=batch_size, callbacks=[EarlyStop])

    # save model
    model_json = model.to_json()
    with open('modelP.json', 'w') as file:
        file.write(model_json)
    model.save_weights('modelW.h5')

    return history


def plot_loss(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='train loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # load data
    X = np.load('X_5855.npy', allow_pickle=True)
    Y = np.load('Y_5855.npy', allow_pickle=True)
    # print(X.shape, Y.shape)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=5)

    model = create_lstm()
    history = train_model(model, X_train, Y_train, X_valid, Y_valid)
    model.summary()

    plot_loss(history)






