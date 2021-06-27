# !pip install --quiet efficientnet

import math, os, re, warnings, random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from data_preparation import *
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras import optimizers, applications, Sequential, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
import efficientnet.tfkeras as efn

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')


# TPU or GPU detection
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU {tpu.master()}')
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


BATCH_SIZE = 16 * REPLICAS
LEARNING_RATE = 3e-5 * REPLICAS

DEBUG=False

if DEBUG:
    EPOCHS = 1
    N_FOLDS = 2
    HEIGHT = 64
    WIDTH = 64
else:
    EPOCHS = 40
    N_FOLDS = 5    
    HEIGHT = 384
    WIDTH = 384


#For CPU Trial
# HEIGHT = 128
# WIDTH = 128

CHANNELS = 3
N_CLASSES = 5

def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)


database_base_path = '/kaggle/input/cassava-leaf-disease-classification/'
train = pd.read_csv(f'{database_base_path}train.csv')
print(f'Train samples: {len(train)}')

GCS_PATH = KaggleDatasets().get_gcs_path('cassava-leaf-disease-classification') 

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train_tfrecords/*.tfrec') 

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

print(f'GCS: train images: {NUM_TRAINING_IMAGES}')
display(train.head())

CLASSES = ['Cassava Bacterial Blight', 
           'Cassava Brown Streak Disease', 
           'Cassava Green Mottle', 
           'Cassava Mosaic Disease', 
           'Healthy']


# Model evaluation
def plot_metrics(history):
    metric_list = [m for m in list(history.keys()) if m is not 'lr']
    size = len(metric_list)//2
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 4))
    if size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for index in range(len(metric_list)//2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)
        if 'loss' in metric_name:
            axes[index].axvline(np.argmin(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmin(history[val_metric_name]), linestyle='dashed', color='orange')
        else:
            axes[index].axvline(np.argmax(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmax(history[val_metric_name]), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
def get_lr_callback():
    lr_start   = 0.000001
    lr_max     = 0.000005 * BATCH_SIZE
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max    
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min    
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    return lr_callback

weight_path_save = 'best_model.hdf5'
last_weight_path = 'last_model.hdf5'

checkpoint = ModelCheckpoint(weight_path_save, 
                             monitor= 'val_sparse_categorical_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode= 'max', 
                             save_weights_only = False)
checkpoint_last = ModelCheckpoint(last_weight_path, 
                             monitor= 'val_sparse_categorical_accuracy', 
                             verbose=1, 
                             save_best_only=False, 
                             mode= 'max', 
                             save_weights_only = False)


early_stopping = EarlyStopping(monitor= 'val_sparse_categorical_accuracy', 
                      mode= 'max', 
                      min_delta=0.1,
                      patience=5,
                      restore_best_weights=True,
                      verbose=1)

LR_scheduler= get_lr_callback()
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.00001)

# Loss Function
# Bi Tempered Logistic Loss

def log_t(u, t):
    epsilon = 1e-7
    """Compute log_t for `u`."""
    if t == 1.0:
        return tf.math.log(u + epsilon)
    else:
        return (u**(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u`."""
    if t == 1.0:
        return tf.math.exp(u)
    else:
        return tf.math.maximum(0.0, 1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

def bi_tempered_logistic_loss( y_true,y_pred, t1, label_smoothing=0.0):
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    y_pred: A multi-dimensional probability tensor with last dimension `num_classes`.
    y_true: A tensor with shape and dtype as y_pred.
    t1: Temperature 1 (< 1.0 for boundedness).
    label_smoothing: A float in [0, 1] for label smoothing.
    Returns:
    A loss tensor.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    if label_smoothing > 0.0:
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = (1 - num_classes /(num_classes - 1) * label_smoothing) * y_true + label_smoothing / (num_classes - 1)

    temp1 = (log_t(y_true + 1e-7, t1) - log_t(y_pred, t1)) * y_true
    temp2 = (1 / (2 - t1)) * (tf.math.pow(y_true, 2 - t1) - tf.math.pow(y_pred, 2 - t1))
    loss_values = temp1 - temp2

    return tf.math.reduce_sum(loss_values, -1)

class BiTemperedLogisticLoss(tf.keras.losses.Loss):
    def __init__(self, t1=0.2, label_smoothing=0.1):
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        return bi_tempered_logistic_loss(y_pred, y_true, self.t1, self.label_smoothing)
    
with strategy.scope():
    def model_fn_version_1(input_shape,N_CLASSES,base_model):
        input_image = L.Input(shape=input_shape, name='input_image')
        model = Sequential([
                        base_model,
                        L.Dropout(.25),
                        L.Dense(32),
                        L.Dense(N_CLASSES, activation='softmax', name='output')
                    ])

        optimizer = optimizers.Adam(lr=LEARNING_RATE)
        metrics=['sparse_categorical_accuracy','accuracy']

        label_smoothing = 0.1
        t1=0.2
        smoothed_btll=BiTemperedLogisticLoss(t1=t1, label_smoothing=label_smoothing)
        
        loss_function= smoothed_btll#tf.keras.losses.SparseCategoricalCrossentropy()

        model.compile(optimizer=optimizer, 
                      loss=loss_function,
                      metrics=metrics)


        return model


    def model_fn_version_2(input_shape,N_CLASSES,base_model):
        input_image = L.Input(shape=input_shape, name='input_image')
        model = Sequential([
                        base_model,
                        L.Dropout(.25),
                        L.Dense(N_CLASSES, activation='softmax', name='output')
                    ])

        optimizer = optimizers.Adam(lr=LEARNING_RATE)
        metrics=['sparse_categorical_accuracy','accuracy']

        label_smoothing = 0.1
        t1=0.2
        smoothed_btll=BiTemperedLogisticLoss(t1=t1, label_smoothing=label_smoothing)
        
        loss_function= smoothed_btll#tf.keras.losses.SparseCategoricalCrossentropy()

        model.compile(optimizer=optimizer, 
                      loss=loss_function,
                      metrics=metrics)


        return model


# EFFICIENT_NET_B4 TRAINING


skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
oof_pred = []; oof_labels = []; history_list = []

for fold,(idxT, idxV) in enumerate(skf.split(np.arange(16))):
    if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {idxT} VALID: {idxV}')

    # Create train and validation sets
    FILENAMES= tf.io.gfile.glob([GCS_PATH + '/train_tfrecords/*.tfrec'])
    TRAIN_FILENAMES = []
    VALID_FILENAMES = []

    for x in idxT:
        TRAIN_FILENAMES.append(FILENAMES[x])

    for x in idxV:
        VALID_FILENAMES.append(FILENAMES[x])

    ## MODEL
    K.clear_session()

    model_path="model_eff_b4_{}.h5"

    with strategy.scope():
        base_model_efficient_net_b4=efn.EfficientNetB4(input_tensor=None,weights=None,include_top=False,pooling='avg')
        model = model_fn_version_1((None, None, CHANNELS), N_CLASSES,base_model_efficient_net_b4)


    model_path = model_path.format(fold)     
    train_data=get_dataset(TRAIN_FILENAMES, labeled=True, ordered=False, repeated=True, augment=True)
    valid_data=get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)

    callbacks_list= [checkpoint, checkpoint_last, early_stopping, LR_scheduler]
    ct_train = count_data_items(TRAIN_FILENAMES)

    ## TRAIN
    history = model.fit(x=train_data, 
                        validation_data=valid_data, 
                        steps_per_epoch=(ct_train // BATCH_SIZE), 
                        callbacks=callbacks_list, 
                        epochs=EPOCHS,  
                        verbose=1).history

    history_list.append(history)
    # Save last model weights
    model.save_weights(model_path)

# OOF predictions
ds_valid = get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)
oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
x_oof = ds_valid.map(lambda image, image_name: image)
oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))

## RESULTS
print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")


y_true = np.concatenate(oof_labels)
y_preds = np.concatenate(oof_pred)

print(classification_report(y_true, y_preds, target_names=CLASSES))



# for fold, history in enumerate(history_list):
#     print(f'\nFOLD: {fold+1}')
#     plot_metrics(history)

# fig, ax = plt.subplots(1, 1, figsize=(20, 12))
# train_cfn_matrix = confusion_matrix(y_true, y_preds, labels=range(len(CLASSES)))
# train_cfn_matrix = (train_cfn_matrix.T / train_cfn_matrix.sum(axis=1)).T
# train_df_cm = pd.DataFrame(train_cfn_matrix, index=CLASSES, columns=CLASSES)
# ax = sns.heatmap(train_df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('Train', fontsize=30)
# plt.show()







# EFFICIENT_NET_B0 TRAINING

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
oof_pred = []; oof_labels = []; history_list = []

for fold,(idxT, idxV) in enumerate(skf.split(np.arange(16))):
    if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {idxT} VALID: {idxV}')

    # Create train and validation sets
    FILENAMES= tf.io.gfile.glob([GCS_PATH + '/train_tfrecords/*.tfrec'])
    TRAIN_FILENAMES = []
    VALID_FILENAMES = []

    for x in idxT:
        TRAIN_FILENAMES.append(FILENAMES[x])

    for x in idxV:
        VALID_FILENAMES.append(FILENAMES[x])

    ## MODEL
    K.clear_session()

    model_path="model_eff_b0_{}.h5"

    with strategy.scope():
        base_model_efficient_net_b0=efn.EfficientNetB0(input_tensor=None,weights=None,include_top=False,pooling='avg')
        model = model_fn_version_2((None, None, CHANNELS), N_CLASSES,base_model_efficient_net_b0)


    model_path = model_path.format(fold)     
    train_data=get_dataset(TRAIN_FILENAMES, labeled=True, ordered=False, repeated=True, augment=True)
    valid_data=get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)

    callbacks_list= [checkpoint, checkpoint_last, early_stopping, LR_scheduler]
    ct_train = count_data_items(TRAIN_FILENAMES)

    ## TRAIN
    history = model.fit(x=train_data, 
                        validation_data=valid_data, 
                        steps_per_epoch=(ct_train // BATCH_SIZE), 
                        callbacks=callbacks_list, 
                        epochs=EPOCHS,  
                        verbose=1).history

    history_list.append(history)
    # Save last model weights
    model.save_weights(model_path)

# OOF predictions
ds_valid = get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)
oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
x_oof = ds_valid.map(lambda image, image_name: image)
oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))

## RESULTS
print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")

y_true = np.concatenate(oof_labels)
y_preds = np.concatenate(oof_pred)

print(classification_report(y_true, y_preds, target_names=CLASSES))


# for fold, history in enumerate(history_list):
#     print(f'\nFOLD: {fold+1}')
#     plot_metrics(history)

# fig, ax = plt.subplots(1, 1, figsize=(20, 12))
# train_cfn_matrix = confusion_matrix(y_true, y_preds, labels=range(len(CLASSES)))
# train_cfn_matrix = (train_cfn_matrix.T / train_cfn_matrix.sum(axis=1)).T
# train_df_cm = pd.DataFrame(train_cfn_matrix, index=CLASSES, columns=CLASSES)
# ax = sns.heatmap(train_df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('Train', fontsize=30)
# plt.show()







# RESNET101 TRAINING

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
oof_pred = []; oof_labels = []; history_list = []

for fold,(idxT, idxV) in enumerate(skf.split(np.arange(16))):
    if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {idxT} VALID: {idxV}')

    # Create train and validation sets
    FILENAMES= tf.io.gfile.glob([GCS_PATH + '/train_tfrecords/*.tfrec'])
    TRAIN_FILENAMES = []
    VALID_FILENAMES = []

    for x in idxT:
        TRAIN_FILENAMES.append(FILENAMES[x])

    for x in idxV:
        VALID_FILENAMES.append(FILENAMES[x])

    ## MODEL
    K.clear_session()

    model_path="model_resnet_101_{}.h5"
    
    with strategy.scope():
        base_model_resnet101=tf.keras.applications.ResNet101(weights=None,include_top=False,input_tensor=None,pooling='avg')
        model = model_fn_version_2((None, None, CHANNELS), N_CLASSES,base_model_resnet101)


    model_path = model_path.format(fold)     
    train_data=get_dataset(TRAIN_FILENAMES, labeled=True, ordered=False, repeated=True, augment=True)
    valid_data=get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)

    callbacks_list= [checkpoint, checkpoint_last, early_stopping, LR_scheduler]
    ct_train = count_data_items(TRAIN_FILENAMES)

    ## TRAIN
    history = model.fit(x=train_data, 
                        validation_data=valid_data, 
                        steps_per_epoch=(ct_train // BATCH_SIZE), 
                        callbacks=callbacks_list, 
                        epochs=EPOCHS,  
                        verbose=1).history

    history_list.append(history)
    # Save last model weights
    model.save_weights(model_path)

# OOF predictions
ds_valid = get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)
oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
x_oof = ds_valid.map(lambda image, image_name: image)
oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))

## RESULTS
print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")


y_true = np.concatenate(oof_labels)
y_preds = np.concatenate(oof_pred)

print(classification_report(y_true, y_preds, target_names=CLASSES))



# for fold, history in enumerate(history_list):
#     print(f'\nFOLD: {fold+1}')
#     plot_metrics(history)

# fig, ax = plt.subplots(1, 1, figsize=(20, 12))
# train_cfn_matrix = confusion_matrix(y_true, y_preds, labels=range(len(CLASSES)))
# train_cfn_matrix = (train_cfn_matrix.T / train_cfn_matrix.sum(axis=1)).T
# train_df_cm = pd.DataFrame(train_cfn_matrix, index=CLASSES, columns=CLASSES)
# ax = sns.heatmap(train_df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('Train', fontsize=30)
# plt.show()


# DID NT USE MODEL TRAINING AS A FUNCTION DUE TO FailedPreconditionError: SharedMemoryAddress expired. ERROR 
# def model_training(model,model_path):
#     skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
#     oof_pred = []; oof_labels = []; history_list = []

#     for fold,(idxT, idxV) in enumerate(skf.split(np.arange(16))):
#         if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
#         print(f'\nFOLD: {fold+1}')
#         print(f'TRAIN: {idxT} VALID: {idxV}')

#         # Create train and validation sets
#         FILENAMES= tf.io.gfile.glob([GCS_PATH + '/train_tfrecords/*.tfrec'])
#         TRAIN_FILENAMES = []
#         VALID_FILENAMES = []

#         for x in idxT:
#             TRAIN_FILENAMES.append(FILENAMES[x])

#         for x in idxV:
#             VALID_FILENAMES.append(FILENAMES[x])

#         ## MODEL
#         K.clear_session()

#     #     model_path="model_eff_b4_{}.h5"

#     #     with strategy.scope():
#     #         base_model_efficient_net_b4=efn.EfficientNetB4(input_tensor=None,weights=None,include_top=False,pooling='avg')
#     #         model = model_fn_version_1((None, None, CHANNELS), N_CLASSES,base_model_efficient_net_b4)


#         model_path = model_path.format(fold)     
#         train_data=get_dataset(TRAIN_FILENAMES, labeled=True, ordered=False, repeated=True, augment=True)
#         valid_data=get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)

#         callbacks_list= [checkpoint, checkpoint_last, early_stopping, LR_scheduler]
#         ct_train = count_data_items(TRAIN_FILENAMES)

#         ## TRAIN
#         history = model.fit(x=train_data, 
#                             validation_data=valid_data, 
#                             steps_per_epoch=(ct_train // BATCH_SIZE), 
#                             callbacks=callbacks_list, 
#                             epochs=EPOCHS,  
#                             verbose=1)

#         history_list.append(history)
#         # Save last model weights
#         model.save_weights(model_path)

#     # OOF predictions
#     ds_valid = get_dataset(VALID_FILENAMES, labeled=True, ordered=True, repeated=False, augment=False)
#     oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
#     x_oof = ds_valid.map(lambda image, image_name: image)
#     oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))

#     ## RESULTS
#     print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_sparse_categorical_accuracy']):.3f}")

#     for fold, history in enumerate(history_list):
#         print(f'\nFOLD: {fold+1}')
#         plot_metrics(history)



#     y_true = np.concatenate(oof_labels)
#     y_preds = np.concatenate(oof_pred)

#     print(classification_report(y_true, y_preds, target_names=CLASSES))


#     fig, ax = plt.subplots(1, 1, figsize=(20, 12))
#     train_cfn_matrix = confusion_matrix(y_true, y_preds, labels=range(len(CLASSES)))
#     train_cfn_matrix = (train_cfn_matrix.T / train_cfn_matrix.sum(axis=1)).T
#     train_df_cm = pd.DataFrame(train_cfn_matrix, index=CLASSES, columns=CLASSES)
#     ax = sns.heatmap(train_df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('Train', fontsize=30)
#     plt.show()



# # EFFICIENT_NET_B4
# model_path="model_eff_b4_{}.h5"

# with strategy.scope():
#     base_model_efficient_net_b4=efn.EfficientNetB4(input_tensor=None,weights=None,include_top=False,pooling='avg')
#     model = model_fn_version_1((None, None, CHANNELS), N_CLASSES,base_model_efficient_net_b4)

# model_training(model,model_path)

# # EFFICIENT_NET_B0
# model_path="model_eff_b0_{}.h5"

# with strategy.scope():
#     base_model_efficient_net_b0=efn.EfficientNetB0(input_tensor=None,weights=None,include_top=False,pooling='avg')
#     model = model_fn_version_2((None, None, CHANNELS), N_CLASSES,base_model_efficient_net_b0)

# model_training(model,model_path)

# # RESNET101
# model_path="model_resnet_101_{}.h5"
# with strategy.scope():
#     base_model_resnet101=tf.keras.applications.ResNet101(weights=None,include_top=False,input_tensor=None,pooling='avg')
#     model = model_fn_version_2((None, None, CHANNELS), N_CLASSES,base_model_resnet101)
    