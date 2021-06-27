# !python -m pip install --quiet --upgrade pip
# !pip install --quiet /kag|gle/input/kerasapplications
# !pip install --quiet /kaggle/input/efficientnet-git


import math, os, re, warnings, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import glob
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, applications, Sequential, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa

# Seeding the variable for consistent results
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



# Basic variable values

BATCH_SIZE = 128 * REPLICAS
HEIGHT = 512
WIDTH = 512 
CHANNELS = 3
N_CLASSES = 5
NO_OF_FOLDS=5


# to perform TTA if > 0 
TTA_STEPS = 3 

# Data preparation functions (AUGMENTATIONS+DATASET CREATION)
def flip_lr(images):
    return tf.image.random_flip_left_right(images)

def flip_ud(images):
    return tf.image.flip_up_down(images)

def shift(images, shift=-3, axis=1):
    return tfa.image.translate(images,[45,37])

def rotate1(images):
    return tfa.image.rotate(images,angles=30)

def rotate2(images):
    return tfa.image.rotate(images,angles=-30)


# Datasets utility functions
def get_name(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    name = parts[-1]
    return name

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def resize_image(image, label):
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])
    return image, label

def process_path(file_path):
    name = get_name(file_path)
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, name

def get_dataset(files_path, shuffled=False, extension='jpg'):
    dataset = tf.data.Dataset.list_files(f'{files_path}*{extension}', shuffle=shuffled)
    dataset = dataset.map(process_path, num_parallel_calls=AUTO)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# function which creates model 
# Type 1
def model_fn_version_1(input_shape,N_CLASSES,base_model):
    input_image = L.Input(shape=input_shape, name='input_image')
    model = Sequential([
                    base_model,
                    L.Dropout(.25),
                    L.Dense(32),
                    L.Dense(N_CLASSES, activation='softmax', name='output')
                ])

    return model

# Type 2
def model_fn_version_2(input_shape,N_CLASSES,base_model):
    input_image = L.Input(shape=input_shape, name='input_image')
    model = Sequential([
                    base_model,
                    L.Dropout(.25),
                    L.Dense(N_CLASSES, activation='softmax', name='output')
                ])

    return model

# function which generates prediction
def generate_prediction(model,file_path,test_pred,tta_functions,TTA_STEPS,N_CLASSES,NO_OF_FOLDS):
    
    rand = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if rand >=0.4 and TTA_STEPS >0:
        no_of_tta=len(tta_functions) 
        tta_pred=np.zeros((len(os.listdir(file_path)),N_CLASSES))
        for func in tta_functions:
            ds=get_dataset(model_files_path)
            ds.map(lambda image,image_name:func(image))
            images= ds.map(lambda image, image_name: image)
            tta_pred+=(model.predict(images,verbose=1))/no_of_tta
        
        test_pred+=(tta_pred/NO_OF_FOLDS)
        
    else:     
        ds=get_dataset(model_files_path)
        images= ds.map(lambda image, image_name: image)
        test_pred+=(model.predict(images,verbose=1)/NO_OF_FOLDS)
    
    return test_pred

# function for weighted prediction blending
def blending_predictions(final_preds,preds_list,weights_list):
    NO_OF_PREDICTIONS=len(preds_list)
    SUM_OF_WEIGHTS=np.sum(weights_list)
    
    for index in range(NO_OF_PREDICTIONS):
        final_preds+=(preds_list[index]*weights_list[index])/NO_OF_PREDICTIONS
        
    return final_preds


database_base_path='../input/cassava-leaf-disease-classification/'
TEST_FILENAMES = tf.io.gfile.glob(f'{database_base_path}test_tfrecords/ld_test*.tfrec')
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
test_list = glob.glob(database_base_path+'/test_images/*.jpg')

NO_OF_FOLDS=5
TTA_STEPS=3

tta_functions=[flip_lr,flip_ud,shift,rotate1,rotate2]

# Train files path
# files_path = database_base_path+'train_images/'
# model_files_path = database_base_path+'train_images/'

# Test files path
files_path = database_base_path+'test_images/'
model_files_path = database_base_path+'test_images/'

# EfficientNetB4 Model predictions
TTA_STEPS=3
test_pred_efficient_net_b4= np.zeros((len(os.listdir(files_path)),N_CLASSES))
base_model_efficient_net_b4=efn.EfficientNetB4(input_tensor=None,weights=None,include_top=False,pooling='avg')
# print(base_model_efficient_net_b4.summary())

for i in range(NO_OF_FOLDS):

    model=model_fn_version_1((HEIGHT,WIDTH,3),N_CLASSES,base_model_efficient_net_b4)
    model.load_weights(f'../input/cassava-data-files/model_eff_b4_{i}.h5')
    
    test_pred_efficient_net_b4+=generate_prediction(model,files_path,test_pred_efficient_net_b4,tta_functions,TTA_STEPS,N_CLASSES,NO_OF_FOLDS)
    

# EfficientNetB0 Model predictions
TTA_STEPS=3
test_pred_efficient_net_b0= np.zeros((len(os.listdir(files_path)),N_CLASSES))
base_model_efficient_net_b0=efn.EfficientNetB0(input_tensor=None,weights=None,include_top=False,pooling='avg')
# print(base_model_efficient_net_b0.summary())

for i in range(NO_OF_FOLDS):

    model=model_fn_version_2((HEIGHT,WIDTH,3),N_CLASSES,base_model_efficient_net_b0)
    # model.load_weights(f'../input/cassava-leaf-disease-tpu-tensorflow-tra-86a840/model_{i}.h5')
    model.load_weights(f'../input/cassava-leaf-disease-tpu-tensorflow-tra-86a840/model_eff_b0_{i}.h5')

    
    test_pred_efficient_net_b0+=generate_prediction(model,files_path,test_pred_efficient_net_b0,tta_functions,TTA_STEPS,N_CLASSES,NO_OF_FOLDS)
# print(base_model_efficient_net_b0.summary())


# ResNet101 Model predictions
TTA_STEPS=3
test_pred_resnet101= np.zeros((len(os.listdir(files_path)),N_CLASSES))
base_model_resnet101=tf.keras.applications.ResNet101(weights=None,include_top=False,input_tensor=None,pooling='avg')
# print(base_model_resnet101.summary())

for i in range(NO_OF_FOLDS):

    model=model_fn_version_2((HEIGHT,WIDTH,3),N_CLASSES,base_model_resnet101)
    model.load_weights(f'../input/cassava-data-files/model_resnet101_{i}.h5')
    
    test_pred_resnet101+=generate_prediction(model,files_path,test_pred_resnet101,tta_functions,TTA_STEPS,N_CLASSES,NO_OF_FOLDS)


# Final prediction generation
test_pred= np.zeros((len(os.listdir(files_path)),N_CLASSES))
list_of_predictions=[test_pred_efficient_net_b4,test_pred_efficient_net_b0,test_pred_resnet101]
list_of_weights=np.ones(3)

test_pred=blending_predictions(test_pred,list_of_predictions,list_of_weights)
final_pred=np.argmax(test_pred, axis=-1)

# Final submission file
ds=get_dataset(model_files_path)
ds_ids = ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(ds_ids.batch(NUM_TEST_IMAGES))).numpy().astype('U')
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, final_pred]), fmt=['%s', '%d'], delimiter=',', header='image_id,label', comments='')   



base_path="../input/cassava-leaf-disease-classification/"
with open(os.path.join(base_path,"label_num_to_disease_map.json"),'r')  as f:
    classes=json.load(f)
    classes={ int(k) : v for k,v in classes.items() }

diseases=list(classes.values())
train=pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
true_pred=train.label.values

y_true=true_pred
y_preds=final_pred


print("\n\n")
print("Model Performance:\n")
print(f"Accuracy: {round(accuracy_score(y_true,y_preds),3)} \n")
print("Classification Report ( Precision, Recall and F1-Score) :\n")
print(classification_report(y_true, y_preds, target_names=diseases))
print("\n\n")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
train_cfn_matrix = confusion_matrix(y_true, y_preds, labels=range(len(diseases)))
train_cfn_matrix = (train_cfn_matrix.T / train_cfn_matrix.sum(axis=1)).T
train_df_cm = pd.DataFrame(train_cfn_matrix, index=diseases, columns=diseases)
ax = sns.heatmap(train_df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('Model Performance: Confusion Matrix', fontsize=24)
plt.show()