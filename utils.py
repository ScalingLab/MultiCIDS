import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def csv_header(file_path):
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)  # Read the first row

            if any(is_numeric(item) for item in first_row):  # If any item is numeric
                print(f"WARNING: The first row in the CSV file {file_path} contains numeric values. Assuming no header.")
                return [] # Return an empty list when there is no header
            else:
                if len(first_row) == len(set(first_row)):  # Check for duplicates
                    return first_row
                else:
                    print(f"WARNING: Duplicates in the first row of file {file_path}. Assuming no header.")
                    return []
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found.")
        return None # Return None when the file is not found

def is_numeric(s):
    """Check if a string can be converted to a float or integer."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def check_label_presence(names):
    if 'Label' not in names:
        print("The 'Label' feature is missing.")
        return False
    return True

def transform_data(training, val_test, has_header, names, features):
    if has_header:
        dfTrain = pd.read_csv(training, names=names, header=0, sep=',', index_col=False, dtype='unicode') # header=0 means the first row is the header
        dfValTest = pd.read_csv(val_test, names=names, header=0, sep=',', index_col=False, dtype='unicode')
    else:
        dfTrain = pd.read_csv(training, names=names, header=None, sep=',', index_col=False, dtype='unicode') # header=None means there is no header
        dfValTest = pd.read_csv(val_test, names=names, header=None, sep=',', index_col=False, dtype='unicode')

    # Store the original labels (useful if they include the attack type)
    dfTrain['OriginalLabel'] = dfTrain['Label']
    dfValTest['OriginalLabel'] = dfValTest['Label']
    
    dfTrain['Label'] = dfTrain['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')
    dfValTest['Label'] = dfValTest['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')

    # Get X, Y, L and O (text labels)
    XTrain, YTrain, LTrain, OTrain = getXY(dfTrain, features)
    XValTest, YValTest, LValTest, OValTest = getXY(dfValTest, features)

    scaler = MinMaxScaler()

    # Check that all values in XTrain and XValTest are finite
    if not np.all(np.isfinite(XTrain)):
        print("WARNING: XTrain contains non-finite values. Checking for infinities and NaNs...")
        if np.any(np.isinf(XTrain)):
            print("WARNING: XTrain contains infinities.")
            inf_indices = np.where(np.isinf(XTrain))
            print("Infinities found at indices:", inf_indices)
        if np.any(np.isnan(XTrain)):
            print("WARNING: XTrain contains NaNs.")
            nan_indices = np.where(np.isnan(XTrain))
            print("NaNs found at indices:", nan_indices)
    else:
        XTrain = scaler.fit_transform(XTrain)
    
    if not np.all(np.isfinite(XValTest)):
        print("WARNING: XValTest contains non-finite values. Checking for infinities and NaNs...")
        if np.any(np.isinf(XValTest)):
            print("WARNING: XValTest contains infinities.")
            inf_indices = np.where(np.isinf(XValTest))
            print("Infinities found at indices:", inf_indices)
        if np.any(np.isnan(XValTest)):
            print("WARNING: XValTest contains NaNs.")
            nan_indices = np.where(np.isnan(XValTest))
            print("NaNs found at indices:", nan_indices)
    else:
        XValTest = scaler.transform(XValTest)

    return XTrain, YTrain, LTrain, OTrain, XValTest, YValTest, LValTest, OValTest

def transform_data_rnn(training, val_test, has_header, names, features, timesteps):
    if has_header:
        dfTrain = pd.read_csv(training, names=names, header=0, sep=',', index_col=False, dtype='unicode') # header=0 means the first row is the header
        dfValTest = pd.read_csv(val_test, names=names, header=0, sep=',', index_col=False, dtype='unicode')
    else:
        dfTrain = pd.read_csv(training, names=names, header=None, sep=',', index_col=False, dtype='unicode') # header=None means there is no header
        dfValTest = pd.read_csv(val_test, names=names, header=None, sep=',', index_col=False, dtype='unicode')

    # Store the original labels (useful if they include the attack type)
    dfTrain['OriginalLabel'] = dfTrain['Label']
    dfValTest['OriginalLabel'] = dfValTest['Label']
    
    dfTrain['Label'] = dfTrain['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')
    dfValTest['Label'] = dfValTest['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')

    # Get X, Y, L and O (text labels)
    XTrain, YTrain, LTrain, OTrain = getXY(dfTrain, features)
    XValTest, YValTest, LValTest, OValTest = getXY(dfValTest, features)

    scaler = MinMaxScaler()

    # Check that all values in XTrain and XValTest are finite
    if not np.all(np.isfinite(XTrain)):
        print("WARNING: XTrain contains non-finite values. Checking for infinities and NaNs...")
        if np.any(np.isinf(XTrain)):
            print("WARNING: XTrain contains infinities.")
            inf_indices = np.where(np.isinf(XTrain))
            print("Infinities found at indices:", inf_indices)
        if np.any(np.isnan(XTrain)):
            print("WARNING: XTrain contains NaNs.")
            nan_indices = np.where(np.isnan(XTrain))
            print("NaNs found at indices:", nan_indices)
    else:
        XTrain = scaler.fit_transform(XTrain)
    
    if not np.all(np.isfinite(XValTest)):
        print("WARNING: XValTest contains non-finite values. Checking for infinities and NaNs...")
        if np.any(np.isinf(XValTest)):
            print("WARNING: XValTest contains infinities.")
            inf_indices = np.where(np.isinf(XValTest))
            print("Infinities found at indices:", inf_indices)
        if np.any(np.isnan(XValTest)):
            print("WARNING: XValTest contains NaNs.")
            nan_indices = np.where(np.isnan(XValTest))
            print("NaNs found at indices:", nan_indices)
    else:
        XValTest = scaler.transform(XValTest)

    # Create sequences
    XTrain, YTrain, LTrain, OTrain = create_sequences(XTrain, YTrain, LTrain, OTrain, timesteps)
    XValTest, YValTest, LValTest, OValTest = create_sequences(XValTest, YValTest, LValTest, OValTest, timesteps)

    print("XTrain shape: ", XTrain.shape)
    print("YTrain shape: ", YTrain.shape)
    print("LTrain shape: ", LTrain.shape)
    print("OTrain shape: ", OTrain.shape)

    return XTrain, YTrain, LTrain, OTrain, XValTest, YValTest, LValTest, OValTest

def getXY(inDataframe, features):
    X = inDataframe[features].values.astype (float)
    colA = np.where(inDataframe['Label']=="BENIGN", 1, 0)
    colB = np.where(inDataframe['Label']=="BENIGN", 0, 1)

    Y = np.column_stack((colA, colB))

    L = inDataframe['Label'].values
    O = inDataframe['OriginalLabel'].values

    return X, Y, L, O

# This function assumes that the data is ordered in time
def create_sequences(X, Y, L, O, timesteps):
    X = np.array(X)
    Y = np.array(Y)
    L = np.array(L)
    O = np.array(O)

    newX = []
    newY = []
    newL = []
    newO = []

    for i in range(len(X) - timesteps + 1):
        newX.append(X[i: i + timesteps])
        newY.append(Y[i + timesteps - 1])
        newL.append(L[i + timesteps - 1])
        newO.append(O[i + timesteps - 1])

    newX = np.array(newX)
    newY = np.array(newY)
    newL = np.array(newL)
    newO = np.array(newO)

    return newX, newY, newL, newO

def detect_subsequence_attacks(window_size=10, tolerance=3):
        attacks = []
        attack_start = None
        consecutive_zeros = 0

        try:
            with open('binary_outcomes.txt', 'r') as f:
                lines = [int(line) for line in f.readlines()]
                if len(lines) < window_size:
                    print("ERROR: File contains less than window_size lines.")
                    return

                window = lines[:window_size]  # Initialize the list with the first window
                ones = np.count_nonzero(window)
                zeros = window_size - ones

                for i in range(window_size, len(lines)):
                    window = lines[i-window_size+1:i+1]  # Update the window

                    # Update the counters
                    if window[-1] == 1:
                        ones += 1
                        consecutive_zeros = 0
                    else:
                        zeros += 1
                        consecutive_zeros += 1

                    if window[0] == 1:
                        ones -= 1
                    else:
                        zeros -= 1

                    if ones > zeros:
                        if attack_start is None:
                            attack_start = i - window_size + window.index(1) + 1
                    else:
                        if attack_start is not None and consecutive_zeros > tolerance:
                            attack_end = i - window[::-1].index(1)  # End index of the attack sequence
                            if attack_end > attack_start:  # Ensure that the end is after the start
                                # Check if the next attack starts before the current one ends
                                if attacks and attacks[-1][1] >= attack_start:
                                    # Merge the overlapping attacks
                                    attacks[-1] = (attacks[-1][0], attack_end)
                                else:
                                    attacks.append((attack_start, attack_end))
                            attack_start = None
                            consecutive_zeros = 0

                # Handle the last window
                if attack_start is not None:
                    attacks.append((attack_start, i))  # i is the last index of the file

            with open('attacks.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['attack_id', 'start_index', 'end_index'])
                for i, (start, end) in enumerate(attacks, start=1):
                    writer.writerow([i, start, end])

        except FileNotFoundError:
            print("ERROR: File binary_outcomes.txt does not exist.")


def evaluate_performance(outcome, evaluationLabels):
        # outcome: boolean      TRUE    ->  BENIGN
        #                       FALSE   ->  ANOMALY

        # evaluationLabels:     original labels

        eval = pd.DataFrame(data={'prediction':outcome, 'Class':evaluationLabels})

        TP = TN = FP = FN = 0

        print('')
        print('             *** EVALUATION RESULTS ***')
        print('')
        print('        Coverage by attack (positive classes)')
        classes = eval['Class'].unique()
        # Recall by class
        print('%6s %10s %10s' % ('FN','TP', 'recall'))
        for c in classes:
            if c != 'BENIGN':
                A = eval[(eval['prediction'] == True)  & (eval['Class'] == c)].shape[0]
                B = eval[(eval['prediction'] == False) & (eval['Class'] == c)].shape[0]

                print ( '%6d %10d %10.3f %26s' %(A, B, B / (A + B), c) )

                FN = FN + A     # cumulative FN
                TP = TP + B     # cumulative TP
            else:
                TN = eval[(eval['prediction'] == True)  & (eval['Class'] == 'BENIGN')].shape[0]
                FP = eval[(eval['prediction'] == False) & (eval['Class'] == 'BENIGN')].shape[0]

        print('%6s %10s' % ('----', '----'))
        print('%6d %10d %10s' % (FN, TP, 'total'))

        print('')
        print('Confusion matrix:')

        print('%42s' % ('prediction'))
        print('%36s | %14s' % (' | BENIGN (neg.)','ATTACK (pos.)'))
        print('       --------------|---------------|---------------')
        print('%28s  %6d | FP = %9d' % ('BENIGN (neg.) | TN = ', TN, FP))
        print('label  --------------|---------------|---------------')
        print('%28s  %6d | TP = %9d' % ('ATTACK (pos.) | FN = ', FN, TP))
        print('       --------------|---------------|---------------')

        recall = 0
        if TP + FN != 0:
            recall = TP / (TP + FN)
        
        precision = 0
        if TP + FP != 0:
            precision = TP / (TP + FP)
        
        f1 = 0
        if precision + recall != 0:
            f1=2 * ((precision * recall) / (precision + recall))
        
        fpr = 0
        if FP + TN != 0:
            fpr = FP / (FP + TN)

        print('Metrics:')
        print('R = %5.3f  P = %5.3f  F1 score = %5.3f  FPR = %5.3f' % (recall, precision, f1, fpr))

def evaluate_subseq_performance(outcome, evaluationLabels):
    # outcome: boolean      TRUE    ->  BENIGN
    #                       FALSE   ->  ANOMALY

    # evaluationLabels:     original labels

    eval = pd.DataFrame(data={'prediction':outcome, 'Class':evaluationLabels})

    attack_intervals = pd.read_csv('attacks.csv')

    # Create a new column that indicates whether the current index is within an attack interval
    eval['in_attack_interval'] = eval.index.to_series().apply(lambda x: 
        any((attack_intervals['start_index'] <= x) & (attack_intervals['end_index'] >= x)))

    TP = ((eval['prediction'] == False) & (eval['Class'] != 'BENIGN') & eval['in_attack_interval']).sum()
    TN = ((eval['prediction'] == True) & (eval['Class'] == 'BENIGN') & ~eval['in_attack_interval']).sum()
    FP = ((eval['prediction'] == False) & (eval['Class'] == 'BENIGN') & eval['in_attack_interval']).sum()
    FN = ((eval['prediction'] == True) & (eval['Class'] != 'BENIGN') & ~eval['in_attack_interval']).sum()

    print('')
    print('Confusion matrix (subsequence anom. detection):')

    print('%42s' % ('prediction'))
    print('%36s | %14s' % (' | BENIGN (neg.)','ATTACK (pos.)'))
    print('       --------------|---------------|---------------')
    print('%28s  %6d | FP = %9d' % ('BENIGN (neg.) | TN = ', TN, FP))
    print('label  --------------|---------------|---------------')
    print('%28s  %6d | TP = %9d' % ('ATTACK (pos.) | FN = ', FN, TP))
    print('       --------------|---------------|---------------')

    recall = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)
    
    precision = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    
    f1 = 0
    if precision + recall != 0:
        f1=2 * ((precision * recall) / (precision + recall))
    
    fpr = 0
    if FP + TN != 0:
        fpr = FP / (FP + TN)

    print('Metrics:')
    print('R = %5.3f  P = %5.3f  F1 score = %5.3f  FPR = %5.3f' % (recall, precision, f1, fpr))