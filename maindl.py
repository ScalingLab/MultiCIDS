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

# SWaT
hardcoded_names = ['Timestamp','FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203',
                   'FIT201','MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301',
                   'LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402',
                   'FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504',
                   'FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503',
                   'FIT601','P601','P602','P603','Label']

hardcoded_features = ['FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203',
                   'FIT201','MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301',
                   'LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402',
                   'FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504',
                   'FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503',
                   'FIT601','P601','P602','P603']

def main():
    parser = argparse.ArgumentParser(description="Train and validate/test an autoencoder. Please make sure there is a feature named 'Label' in the CSV files.")
    parser.add_argument('train_set', type=str, help="Path to the training CSV file.")
    parser.add_argument('val_test_set', type=str, help="Path to the validation or test CSV file.")

    args = parser.parse_args()

    if args.train_set is None or args.val_test_set is None:
        parser.print_usage()
        return

    if USE_LSTM and USE_GRU:
        print("ERROR: You cannot use both LSTM and GRU at the same time.")
        return

    train_names = csv_header(args.train_set)
    val_test_names = csv_header(args.val_test_set) # Need val/test header names too for the following checks

    if train_names is None or val_test_names is None:
        print("ERROR: Files should exist. Please check the paths and try again.")
        return
    
    if train_names != val_test_names:
        print("ERROR: The headers in the training set and validation/test set are different.")
        return
    
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
        autoencoder = LSTMAutoEncoder(timesteps=TIMESTEPS, input_dim=XTrain.shape[2])

        print("Number of timesteps:", XTrain.shape[1])
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
        autoencoder = AutoEncoder(input_dim=XTrain.shape[1])

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