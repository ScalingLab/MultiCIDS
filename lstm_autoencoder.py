import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rc('font', size=17)
plt.rc('axes', labelsize=17)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)

from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras import regularizers

import time

tf.keras.utils.set_random_seed(33)
tf.config.experimental.enable_op_determinism()

class LSTMAutoEncoder():
    def __init__(self, timesteps, input_dim):
        input_layer = Input(shape=(timesteps, input_dim))
        layer = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(input_layer)
        layer = LSTM(32, activation='relu', return_sequences=False, kernel_regularizer=regularizers.l2(0.00))(layer)
        layer = RepeatVector(timesteps)(layer)
        layer = LSTM(32, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(layer)
        layer = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(layer)
        output_layer = TimeDistributed(Dense(input_dim))(layer)

        self.lstm_autoencoder = Model(inputs = input_layer, outputs = output_layer)
    
    def summary(self, ):
        self.lstm_autoencoder.summary()
    
    def train(self, x, y, timesteps):
        e = 64
        b = 512
        validation_split = 0.1

        self.lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        start_time = time.time()
        history = self.lstm_autoencoder.fit(x, y, epochs=e, batch_size=b, validation_split=validation_split, verbose=2, shuffle=False)
        # Note: setting shuffle=False is important since we are using LSTM
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for training: {elapsed_time} seconds")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('loss.png')

        # Computation of the detection threshold with a percentage
        # of the training set equal to 'validation_split'
        x_thSet = x[x.shape[0]-(int)(x.shape[0]*validation_split):x.shape[0]-1, :]
        self.threshold = self.compute_threshold(x_thSet, timesteps)

        df_history = pd.DataFrame(history.history)
        return df_history
    
    def compute_threshold(self, x_thSet, timesteps):
        x_thSetPredictions = self.lstm_autoencoder.predict(x_thSet)
        mse = self.compute_mse(x_thSet, x_thSetPredictions, timesteps)

        
        threshold = np.percentile(mse, 95)
        #threshold = np.mean(mse) + 2*np.std(mse)

        print("Threshold: ", threshold)
        
        return threshold
    
    def predict(self, x_evaluation, timesteps):
        predictions = self.lstm_autoencoder.predict(x_evaluation)
        RE = self.compute_mse(x_evaluation, predictions, timesteps)
        outcome = RE <= self.threshold

        return outcome
    
    def save_model(self, file_path):
        self.lstm_autoencoder.save(file_path)
    
    def plot_reconstruction_error(self, x_evaluation, evaluationLabels, timesteps):
        predictions = self.lstm_autoencoder.predict(x_evaluation)
        mse = self.compute_mse(x_evaluation, predictions, timesteps)

        trueClass = evaluationLabels != 'BENIGN'
        errors = pd.DataFrame({'reconstruction_error': mse, 'true_class': trueClass})
        groups = errors.groupby('true_class')

        fig, ax = plt.subplots(figsize=(8, 5))
        right = 0
        for name, group in groups:
            if max(group.index) > right: right = max(group.index)

            ax.plot(group.index, group.reconstruction_error, marker = 'o' if int(name) == 0 else 'x', ms = 3, linestyle = '', #alpha = 0.5,
                 label = 'normal' if int(name) == 0 else 'intrusion', color = 'lightgray' if int(name) == 0 else 'blue')
        ax.hlines(self.threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors = 'red', zorder = 100, label = 'threshold',linewidth=3,linestyles='dashed')
        ax.semilogy()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, markerscale=4)
        plt.xlim(left = 0, right = right)
        plt.ylim(bottom=0.00001, top=100000)
        plt.ylabel('RE')
        plt.xlabel('point index')

        # Make every other xticklabel invisible
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        plt.tight_layout()
        plt.savefig('reconstruction_error.png')

    def plot_reconstruction_error_with_attacks(self, x_evaluation, evaluationLabels, timesteps):
        predictions = self.lstm_autoencoder.predict(x_evaluation)
        mse = self.compute_mse(x_evaluation, predictions, timesteps)

        trueClass = evaluationLabels != 'BENIGN'
        errors = pd.DataFrame({'reconstruction_error': mse, 'true_class': trueClass})
        groups = errors.groupby('true_class')

        fig, ax = plt.subplots(figsize=(8, 5))
        right = 0
        for name, group in groups:
            if max(group.index) > right: right = max(group.index)

            ax.plot(group.index, group.reconstruction_error, marker = 'o' if int(name) == 0 else 'x', ms = 3, linestyle = '', #alpha = 0.5,
                 label = 'normal' if int(name) == 0 else 'intrusion', color = 'lightgray' if int(name) == 0 else 'blue')
        ax.hlines(self.threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors = 'red', zorder = 100, label = 'threshold',linewidth=3,linestyles='dashed')
        ax.semilogy()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, markerscale=4)
        plt.xlim(left = 0, right = right)
        plt.ylim(bottom=0.00001, top=100000)
        plt.ylabel('RE')
        plt.xlabel('point index')

        # Read the start and end indices of the attacks
        attacks = pd.read_csv('attacks.csv')
        for _, row in attacks.iterrows():
            start, end = row['start_index'], row['end_index']
            # Add a translucent red block to the plot to highlight the attack
            ax.add_patch(patches.Rectangle((start, 0), end - start, ax.get_ylim()[1], alpha=0.2, color='red', linewidth=0))
        
        # Make every other xticklabel invisible
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        plt.tight_layout()
        plt.savefig('reconstruction_error_with_attacks.png')

    def compute_mse(self, x_evaluation, predictions, timesteps):
        powers = np.power(x_evaluation - predictions, 2)

        #powers_last = powers[:, timesteps-1, :]
        #mse = np.mean(powers_last, axis=1)

        # To compute the mean squared error on the entire sequence, use the following line
        mse = np.mean(powers, axis=(1, 2))
        # instead of the previous two lines

        return mse
    
    def write_binary_predictions(self, x_evaluation, timesteps):
        predictions = self.lstm_autoencoder.predict(x_evaluation)
        RE = self.compute_mse(x_evaluation, predictions, timesteps)
        binary_predictions = np.where(RE > self.threshold, 1, 0)

        with open('binary_outcomes.txt', 'w') as f:
            for binary_prediction in binary_predictions:
                f.write("%s\n" % binary_prediction)
        
        with open('reconstruction_errors.txt', 'w') as f:
            for error in RE:
                f.write("%s\n" % error)
        
        with open('threshold.txt', 'w') as f:
            f.write("%s\n" % self.threshold)