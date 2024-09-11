import argparse
from utils import csv_header, check_label_presence, transform_data, transform_data_rnn, evaluate_performance, evaluate_subseq_performance, detect_subsequence_attacks
from autoencoder import AutoEncoder
from lstm_autoencoder import LSTMAutoEncoder
from gru_autoencoder import GRUAutoEncoder

import time

USE_LSTM = False
USE_GRU = False

TIMESTEPS = 30 # Only used if USE_LSTM or USE_GRU is True
 
SAVE_MODEL = False

PLOT_SUBSEQUENCE_ATKS = True

# N-BaIoT dataset
hardcoded_names = ['MI_dir_L5_weight','MI_dir_L5_mean','MI_dir_L5_variance','MI_dir_L3_weight','MI_dir_L3_mean','MI_dir_L3_variance',
         'MI_dir_L1_weight','MI_dir_L1_mean','MI_dir_L1_variance','MI_dir_L0.1_weight','MI_dir_L0.1_mean','MI_dir_L0.1_variance',
         'MI_dir_L0.01_weight','MI_dir_L0.01_mean','MI_dir_L0.01_variance','H_L5_weight','H_L5_mean','H_L5_variance','H_L3_weight',
         'H_L3_mean','H_L3_variance','H_L1_weight','H_L1_mean','H_L1_variance','H_L0.1_weight','H_L0.1_mean','H_L0.1_variance',
         'H_L0.01_weight','H_L0.01_mean','H_L0.01_variance','HH_L5_weight','HH_L5_mean','HH_L5_std','HH_L5_magnitude','HH_L5_radius',
         'HH_L5_covariance','HH_L5_pcc','HH_L3_weight','HH_L3_mean','HH_L3_std','HH_L3_magnitude','HH_L3_radius','HH_L3_covariance',
         'HH_L3_pcc','HH_L1_weight','HH_L1_mean','HH_L1_std','HH_L1_magnitude','HH_L1_radius','HH_L1_covariance','HH_L1_pcc',
         'HH_L0.1_weight','HH_L0.1_mean','HH_L0.1_std','HH_L0.1_magnitude','HH_L0.1_radius','HH_L0.1_covariance','HH_L0.1_pcc',
         'HH_L0.01_weight','HH_L0.01_mean','HH_L0.01_std','HH_L0.01_magnitude','HH_L0.01_radius','HH_L0.01_covariance','HH_L0.01_pcc',
         'HH_jit_L5_weight','HH_jit_L5_mean','HH_jit_L5_variance','HH_jit_L3_weight','HH_jit_L3_mean','HH_jit_L3_variance',
         'HH_jit_L1_weight','HH_jit_L1_mean','HH_jit_L1_variance','HH_jit_L0.1_weight','HH_jit_L0.1_mean','HH_jit_L0.1_variance',
         'HH_jit_L0.01_weight','HH_jit_L0.01_mean','HH_jit_L0.01_variance','HpHp_L5_weight','HpHp_L5_mean','HpHp_L5_std','HpHp_L5_magnitude',
         'HpHp_L5_radius','HpHp_L5_covariance','HpHp_L5_pcc','HpHp_L3_weight','HpHp_L3_mean','HpHp_L3_std','HpHp_L3_magnitude',
         'HpHp_L3_radius','HpHp_L3_covariance','HpHp_L3_pcc','HpHp_L1_weight','HpHp_L1_mean','HpHp_L1_std','HpHp_L1_magnitude',
         'HpHp_L1_radius','HpHp_L1_covariance','HpHp_L1_pcc','HpHp_L0.1_weight','HpHp_L0.1_mean','HpHp_L0.1_std','HpHp_L0.1_magnitude',
         'HpHp_L0.1_radius','HpHp_L0.1_covariance','HpHp_L0.1_pcc','HpHp_L0.01_weight','HpHp_L0.01_mean','HpHp_L0.01_std','HpHp_L0.01_magnitude',
         'HpHp_L0.01_radius','HpHp_L0.01_covariance','HpHp_L0.01_pcc','Device','Label']

hardcoded_features = ['MI_dir_L5_weight','MI_dir_L5_mean','MI_dir_L5_variance','MI_dir_L3_weight','MI_dir_L3_mean','MI_dir_L3_variance',
         'MI_dir_L1_weight','MI_dir_L1_mean','MI_dir_L1_variance','MI_dir_L0.1_weight','MI_dir_L0.1_mean','MI_dir_L0.1_variance',
         'MI_dir_L0.01_weight','MI_dir_L0.01_mean','MI_dir_L0.01_variance','H_L5_weight','H_L5_mean','H_L5_variance','H_L3_weight',
         'H_L3_mean','H_L3_variance','H_L1_weight','H_L1_mean','H_L1_variance','H_L0.1_weight','H_L0.1_mean','H_L0.1_variance',
         'H_L0.01_weight','H_L0.01_mean','H_L0.01_variance','HH_L5_weight','HH_L5_mean','HH_L5_std','HH_L5_magnitude','HH_L5_radius',
         'HH_L5_covariance','HH_L5_pcc','HH_L3_weight','HH_L3_mean','HH_L3_std','HH_L3_magnitude','HH_L3_radius','HH_L3_covariance',
         'HH_L3_pcc','HH_L1_weight','HH_L1_mean','HH_L1_std','HH_L1_magnitude','HH_L1_radius','HH_L1_covariance','HH_L1_pcc',
         'HH_L0.1_weight','HH_L0.1_mean','HH_L0.1_std','HH_L0.1_magnitude','HH_L0.1_radius','HH_L0.1_covariance','HH_L0.1_pcc',
         'HH_L0.01_weight','HH_L0.01_mean','HH_L0.01_std','HH_L0.01_magnitude','HH_L0.01_radius','HH_L0.01_covariance','HH_L0.01_pcc',
         'HH_jit_L5_weight','HH_jit_L5_mean','HH_jit_L5_variance','HH_jit_L3_weight','HH_jit_L3_mean','HH_jit_L3_variance',
         'HH_jit_L1_weight','HH_jit_L1_mean','HH_jit_L1_variance','HH_jit_L0.1_weight','HH_jit_L0.1_mean','HH_jit_L0.1_variance',
         'HH_jit_L0.01_weight','HH_jit_L0.01_mean','HH_jit_L0.01_variance','HpHp_L5_weight','HpHp_L5_mean','HpHp_L5_std','HpHp_L5_magnitude',
         'HpHp_L5_radius','HpHp_L5_covariance','HpHp_L5_pcc','HpHp_L3_weight','HpHp_L3_mean','HpHp_L3_std','HpHp_L3_magnitude',
         'HpHp_L3_radius','HpHp_L3_covariance','HpHp_L3_pcc','HpHp_L1_weight','HpHp_L1_mean','HpHp_L1_std','HpHp_L1_magnitude',
         'HpHp_L1_radius','HpHp_L1_covariance','HpHp_L1_pcc','HpHp_L0.1_weight','HpHp_L0.1_mean','HpHp_L0.1_std','HpHp_L0.1_magnitude',
         'HpHp_L0.1_radius','HpHp_L0.1_covariance','HpHp_L0.1_pcc','HpHp_L0.01_weight','HpHp_L0.01_mean','HpHp_L0.01_std','HpHp_L0.01_magnitude',
         'HpHp_L0.01_radius','HpHp_L0.01_covariance','HpHp_L0.01_pcc']

def main():
    parser = argparse.ArgumentParser(description="Train and validate/test a LSTM autoencoder using CSVs. Please make sure there is a feature named 'Label' in the CSVs.")
    parser.add_argument('train_set', type=str, help="Path to the training CSV.")
    parser.add_argument('val_test_set', type=str, help="Path to the validation or test CSV.")

    args = parser.parse_args()

    if args.train_set is None or args.val_test_set is None:
        parser.print_usage()
        return

    if USE_LSTM and USE_GRU:
        print("ERROR: You cannot use both LSTM and GRU at the same time.")
        return

    train_names = csv_header(args.train_set)
    val_test_names = csv_header(args.val_test_set) # Need val/test header names too for the following checks

    # None means the file does not exist
    if train_names is None or val_test_names is None:
        print("ERROR: Files should exist. Please check the paths and try again.")
        return
    
    if train_names != val_test_names:
        print("ERROR: The headers in the training set and validation/test set are different.")
        return
    
    # Empty list means no header
    if not train_names or not val_test_names:
        print("The CSV files should have a header. Adding the hardcoded header names...")
        train_names = hardcoded_names
        val_test_names = hardcoded_names
        if not check_label_presence(train_names) or not check_label_presence(val_test_names):
            return
        train_features = hardcoded_features
        has_header = False
    else:
        if not check_label_presence(train_names) or not check_label_presence(val_test_names):
            return
        excluded_features = input("Please enter the features you want to exclude from training, separated by a comma: ").split(',')
        train_features = [feature for feature in train_names if feature not in excluded_features]
        has_header = True
    
    if USE_LSTM:
        print("Using LSTM autoencoder...")
        XTrain, YTrain, LTrain, OTrain, XValTest, YValTest, LValTest, OValTest = transform_data_rnn(args.train_set, args.val_test_set, has_header, train_names, train_features, TIMESTEPS)
        autoencoder = LSTMAutoEncoder(timesteps=TIMESTEPS, input_dim=XTrain.shape[2]) # XTimesteps.shape[2] is the number of features

        print("Number of timesteps:", XTrain.shape[1]) # XTimesteps.shape[1] should be equal to TIMESTEPS
        print("Number of features:", XTrain.shape[2])
    elif USE_GRU:
        print("Using GRU autoencoder...")
        XTrain, YTrain, LTrain, OTrain, XValTest, YValTest, LValTest, OValTest = transform_data_rnn(args.train_set, args.val_test_set, has_header, train_names, train_features, TIMESTEPS)
        autoencoder = GRUAutoEncoder(timesteps=TIMESTEPS, input_dim=XTrain.shape[2])

        print("Number of timesteps:", XTrain.shape[1])
        print("Number of features:", XTrain.shape[2])
    else:
        print("Using classic autoencoder...")
        XTrain, YTrain, LTrain, OTrain, XValTest, YValTest, LValTest, OValTest = transform_data(args.train_set, args.val_test_set, has_header, train_names, train_features)
        autoencoder = AutoEncoder(input_dim=XTrain.shape[1]) # This time XTrain.shape[1] is the number of features

        print("Number of features:", XTrain.shape[1])

    autoencoder.summary()

    if USE_LSTM or USE_GRU:
        autoencoder.train(XTrain, XTrain, TIMESTEPS)
    else:
        autoencoder.train(XTrain, XTrain)

    if SAVE_MODEL:
        if USE_LSTM:
            autoencoder.save_model('lstm_autoencoder.h5')
        elif USE_GRU:
            autoencoder.save_model('gru_autoencoder.h5')
        else:
            autoencoder.save_model('autoencoder.h5')
    
    start_time = time.time()

    if USE_LSTM or USE_GRU:
        outcome = autoencoder.predict(XValTest, TIMESTEPS)
    else:
        outcome = autoencoder.predict(XValTest)

    #print("Outcome shape:", outcome.shape)
    #print("LValTest shape:", LValTest.shape)

    evaluate_performance(outcome, LValTest)

    if USE_LSTM or USE_GRU:
        autoencoder.write_binary_predictions(XValTest, TIMESTEPS)
    else:
        autoencoder.write_binary_predictions(XValTest)

    detect_subsequence_attacks(window_size=TIMESTEPS, tolerance=3)
    # It may be reasonable for window_size to be equal to TIMESTEPS, but it's not mandatory

    evaluate_subseq_performance(outcome, LValTest)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for prediction: {elapsed_time} seconds")

    if USE_LSTM or USE_GRU:
        if PLOT_SUBSEQUENCE_ATKS:
            autoencoder.plot_reconstruction_error_with_attacks(XValTest, LValTest, TIMESTEPS)
        else:
            autoencoder.plot_reconstruction_error(XValTest, LValTest, TIMESTEPS)
    else:
        if PLOT_SUBSEQUENCE_ATKS:
            autoencoder.plot_reconstruction_error_with_attacks(XValTest, LValTest)
        else:
            autoencoder.plot_reconstruction_error(XValTest, LValTest)

if __name__ == "__main__":
    main()