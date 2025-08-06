import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import pywt
from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Input, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import uuid
import logging

# Set up logging
logging.basicConfig(filename='model_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set matplotlib parameters
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True

# Denoising function
def denoise(data): 
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04
    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec

# Data loading and preprocessing
path = '/kaggle/input/mitbih-database/'
window_size = 180
maximum_counting = 10000
classes = ['N', 'S', 'V', 'F', 'Q']
n_classes = len(classes)
count_classes = [0]*n_classes
X = list()
y = list()

filenames = next(os.walk(path))[2]
records = list()
annotations = list()
filenames.sort()

for f in filenames:
    filename, file_extension = os.path.splitext(f)
    if file_extension == '.csv':
        records.append(path + filename + file_extension)
    else:
        annotations.append(path + filename + file_extension)

for r in range(len(records)):
    signals = []
    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_index = -1
        for row in spamreader:
            if row_index >= 0:
                signals.append(int(row[1]))
            row_index += 1
            
    if r == 1:
        plt.title(records[1] + " Wave")
        plt.plot(signals[0:700])
        plt.savefig('wave_original.png')
        plt.close()
        
    signals = denoise(signals)
    if r == 1:
        plt.title(records[1] + " wave after denoised")
        plt.plot(signals[0:700])
        plt.savefig('wave_denoised.png')
        plt.close()
        
    signals = stats.zscore(signals)
    if r == 1:
        plt.title(records[1] + " wave after z-score normalization")
        plt.plot(signals[0:700])
        plt.savefig('wave_normalized.png')
        plt.close()
    
    example_beat_printed = False
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines()
        beat = list()
        for d in range(1, len(data)):
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted)
            pos = int(next(splitted))
            arrhythmia_type = next(splitted)
            if arrhythmia_type in classes:
                arrhythmia_index = classes.index(arrhythmia_type)
                count_classes[arrhythmia_index] += 1
                if window_size <= pos and pos < (len(signals) - window_size):
                    beat = signals[pos-window_size:pos+window_size]
                    if r == 1 and not example_beat_printed:
                        plt.title("A Beat from " + records[1] + " Wave")
                        plt.plot(beat)
                        plt.savefig('example_beat.png')
                        plt.close()
                        example_beat_printed = True
                    X.append(beat)
                    y.append(arrhythmia_index)

print(np.shape(X), np.shape(y))

for i in range(len(X)):
    X[i] = np.append(X[i], y[i])

X_train_df = pd.DataFrame(X)
per_class = X_train_df[X_train_df.shape[1]-1].value_counts()
print(per_class)
plt.figure(figsize=(20,10))
my_circle = plt.Circle((0,0), 0.7, color='white')
plt.pie(per_class, labels=classes, colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green'], autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig('class_distribution_before.png')
plt.close()

# Balance dataset
df_1 = X_train_df[X_train_df[X_train_df.shape[1]-1] == 1]
df_2 = X_train_df[X_train_df[X_train_df.shape[1]-1] == 2]
df_3 = X_train_df[X_train_df[X_train_df.shape[1]-1] == 3]
df_4 = X_train_df[X_train_df[X_train_df.shape[1]-1] == 4]
df_0 = (X_train_df[X_train_df[X_train_df.shape[1]-1] == 0]).sample(n=5000, random_state=42)

df_1_upsample = resample(df_1, replace=True, n_samples=5000, random_state=122)
df_2_upsample = resample(df_2, replace=True, n_samples=5000, random_state=123)
df_3_upsample = resample(df_3, replace=True, n_samples=5000, random_state=124)
df_4_upsample = resample(df_4, replace=True, n_samples=5000, random_state=125)

X_train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

per_class = X_train_df[X_train_df.shape[1]-1].value_counts()
print(per_class)
plt.figure(figsize=(20,10))
my_circle = plt.Circle((0,0), 0.7, color='white')
plt.pie(per_class, labels=classes, colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green'], autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig('class_distribution_after.png')
plt.close()

train, test = train_test_split(X_train_df, test_size=0.20, random_state=42)
print("X_train : ", np.shape(train))
print("X_test  : ", np.shape(test))

target_train = train[train.shape[1]-1]
target_test = test[test.shape[1]-1]
train_y = to_categorical(target_train)
test_y = to_categorical(target_test)
train_y_ml = target_train.astype(int)
test_y_ml = target_test.astype(int)
print(np.shape(train_y), np.shape(test_y))

train_x = train.iloc[:,:train.shape[1]-1].values
test_x = test.iloc[:,:test.shape[1]-1].values
train_x_cnn = train_x.reshape(len(train_x), train_x.shape[1], 1)
test_x_cnn = test_x.reshape(len(test_x), test_x.shape[1], 1)
train_x_flat = train_x
test_x_flat = test_x
print(np.shape(train_x_cnn), np.shape(test_x_cnn))

# Function to plot ROC curves
def plot_roc_curve(y_true, y_score, model_name, classes):
    plt.figure()
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true[:, i] if y_true.ndim > 1 else (y_true == i).astype(int), y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (class {classes[i]}, AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.savefig(f'{model_name}_cm.png')
    plt.close()

# Function to plot loss and accuracy
def plot_loss_accuracy(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_accuracy.png')
    plt.close()

# CNN Model
def create_cnn_model(input_shape=(360, 1)):
    model = tf.keras.Sequential([
        Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model

# DenseNet Model (Adapted for 1D)
def create_densenet_model(input_shape=(360, 1)):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    for _ in range(6):
        x1 = BatchNormalization()(x)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x1 = Conv1D(32, 3, padding='same')(x1)
        x = Concatenate()([x, x1])
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Inception Model (Adapted for 1D)
def create_inception_model(input_shape=(360, 1)):
    inputs = Input(shape=input_shape)
    
    branch1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    
    branch3 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    branch3 = Conv1D(64, 3, padding='same', activation='relu')(branch3)
    
    branch5 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
    branch5 = Conv1D(64, 5, padding='same', activation='relu')(branch5)
    
    branch_pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)
    branch_pool = Conv1D(32, 1, padding='same', activation='relu')(branch_pool)
    
    x = Concatenate()([branch1, branch3, branch5, branch_pool])
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# YOLO-inspired Model (v8, v9, v11 adapted for 1D classification)
def create_yolo_model(input_shape=(360, 1), version='v8'):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    filters = 64 if version == 'v8' else 128 if version == 'v9' else 256
    for _ in range(3 if version == 'v8' else 4 if version == 'v9' else 5):
        x = Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Dictionary of models
deep_models = {
    'CNN': create_cnn_model,
    'DenseNet': create_densenet_model,
    'Inception': create_inception_model,
    'YOLOv8': lambda: create_yolo_model(version='v8'),
    'YOLOv9': lambda: create_yolo_model(version='v9'),
    'YOLOv11': lambda: create_yolo_model(version='v11')
}

ml_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate deep learning models
for model_name, create_model in deep_models.items():
    try:
        print(f"\nTraining {model_name}...")
        model = create_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        history = model.fit(
            train_x_cnn, train_y,
            batch_size=36,
            epochs=60,
            verbose=1,
            validation_data=(test_x_cnn, test_y)
        )
        
        score = model.evaluate(test_x_cnn, test_y)
        print(f'{model_name} Test Loss: {score[0]}')
        print(f'{model_name} Test Accuracy: {score[1]}')
        
        y_true = np.argmax(test_y, axis=1)
        prediction_proba = model.predict(test_x_cnn)
        prediction = np.argmax(prediction_proba, axis=1)
        
        plot_loss_accuracy(history, model_name)
        plot_roc_curve(test_y, prediction_proba, model_name, classes)
        plot_confusion_matrix(y_true, prediction, model_name, classes)
        
        cf = classification_report(y_true, prediction, target_names=classes, digits=4)
        print(f'\n{model_name} Classification Report:\n{cf}')
        with open(f'{model_name}_classification_report.txt', 'w') as f:
            f.write(cf)
            
    except Exception as e:
        logging.error(f"Error in {model_name}: {str(e)}")
        print(f"Error in {model_name}: {str(e)}")
        continue

# Train and evaluate traditional ML models
for model_name, model in ml_models.items():
    try:
        print(f"\nTraining {model_name}...")
        model.fit(train_x_flat, train_y_ml)
        y_pred = model.predict(test_x_flat)
        y_score = model.predict_proba(test_x_flat) if hasattr(model, 'predict_proba') else to_categorical(y_pred)
        
        accuracy = np.mean(y_pred == test_y_ml)
        print(f'{model_name} Test Accuracy: {accuracy}')
        
        plot_roc_curve(test_y, y_score, model_name, classes)
        plot_confusion_matrix(test_y_ml, y_pred, model_name, classes)
        
        cf = classification_report(test_y_ml, y_pred, target_names=classes, digits=4)
        print(f'\n{model_name} Classification Report:\n{cf}')
        with open(f'{model_name}_classification_report.txt', 'w') as f:
            f.write(cf)
            
    except Exception as e:
        logging.error(f"Error in {model_name}: {str(e)}")
        print(f"Error in {model_name}: {str(e)}")
        continue