import os, sys
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

sys.path.append('../')
from cnn.tf_utils import video_to_frames, SGDRScheduler
from slow_fast_net import SlowFastNet


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
AUTOTUNE = tf.data.experimental.AUTOTUNE


cv2_image = 'cv2_image/cv2.jpg'
video_path = '../../medication/action_recognition/dataset/medi'
video_data = f'{video_path}/hand_main/test_4.mp4'

frame_total = 600
frame_stride = 5

frame_size = 100
video_size = 250
input_size = 224

obj_name = '*'

videos = [path for path in Path(video_path).glob(f'{obj_name}_*/*.mp4')]
none = [path for path in Path(video_path).glob('non/*.mp4')]
videos.extend(none)
for i in range(3000):
    random.shuffle(videos)
video_labels = np.unique(list(map(lambda x:x.parts[-2].split("-")[0],videos)))
video_labels

labels = list(map(lambda x:x.parts[-2].split("-")[0],videos))



label_num = {i: labels.count(i) for i in video_labels}


for i in videos:
    if i.name[0] == '.':
        print(i.name)

resnet_size = 50
input_shape = (frame_size, input_size, input_size, 3)
output_count = len(video_labels)
epoch_size = 120
batch_size = 16


def labels(path):
    path = str(path)
    label_name = tf.strings.split(path,os.path.sep)[-2]
    onehot = tf.cast(video_labels == label_name, tf.uint8)
    return onehot


def raw_to_tensor(data):
    # 프레임 추출
    frame_np = video_to_frames(data, frame_stride, resolution=(video_size,video_size))
    # 프레임 길이 맞추기
    frame_np = same_length(frame_np, frame_total//frame_stride)
    label = labels(data)
    frame_tensor=tf.convert_to_tensor(frame_np, dtype=tf.float32)

    return frame_tensor, label


def same_length(arr, frame_count=frame_total):
    if arr.shape[0] < frame_count:
        added = np.zeros((frame_count-arr.shape[0],*arr.shape[1:]))+1e-5
        arr = np.concatenate([added, arr])
    else:
        arr = arr[:frame_count]
        
    return arr


@tf.function
def image_preprocess(frame_tensor, label):
    frames = tf.cast(frame_tensor/255., tf.float32)
    return frames, label

def iter_datas(start_num, end_num):
    for i in videos[start_num:end_num]:
        (frames, label) = raw_to_tensor(i)
        yield frames, label

@tf.function
def random_contrast(data):
    contrast_fn = partial(tf.image.random_contrast, lower=0.8, upper=1.2)
    result = tf.map_fn(contrast_fn, data, parallel_iterations=AUTOTUNE)
    return result

@tf.function
def random_hue(data):
    hue_fn = partial(tf.image.random_hue, max_delta=0.1)
    result = tf.map_fn(hue_fn, data, parallel_iterations=AUTOTUNE)
    return result

@tf.function
def random_brightness(data):
    brightness_fn = partial(tf.image.random_brightness, max_delta=0.2)
    result = tf.map_fn(brightness_fn, data, parallel_iterations=AUTOTUNE)
    return result

@tf.function
def random_flip_letf_right(data):
    flip_fn = partial(tf.image.random_flip_left_right)
    result = tf.map_fn(flip_fn, data, parallel_iterations=AUTOTUNE)
    return result

@tf.function
def random_crop(data):
    crop_fn = partial(tf.image.random_crop, size=(frame_size,input_size,input_size,3))
    result = tf.map_fn(crop_fn, data, parallel_iterations=AUTOTUNE)
    return result

@tf.function
def random_crop_frames(data, label):
    crop_size = (frame_total//frame_stride)-frame_size
    r = random.randint(0,crop_size)
    result = data[:,r:-(crop_size-r),:,:,:]
    return result, label

@tf.function
def resize(data, label, size=(input_size,input_size)):
    resize_fn = partial(tf.image.resize,size=size)
    result = tf.map_fn(resize_fn, data, parallel_iterations=AUTOTUNE)
    return result, label

@tf.function
def arg_data(data, label):
    random_size = random.randrange(input_size,video_size+1)
    x, _ = resize(data, label, size=(random_size,random_size))
    x, _ = random_crop_frames(x, label)
    x = random_crop(x)
    x = random_contrast(x)
    x = random_hue(x)
    x = random_brightness(x)
    return x, label


train_count = int(len(videos)*0.8)
val_count = len(videos) - train_count
raw_data_shape = tf.TensorShape([frame_total//frame_stride, video_size, video_size, 3])


train_datas = partial(iter_datas,start_num=0, end_num=train_count)
val_datas = partial(iter_datas, start_num=train_count, end_num=-1)

# Train Data
_train_dataset = tf.data.Dataset.from_generator(
            train_datas,
            (tf.float32, tf.uint8),
            (raw_data_shape, tf.TensorShape([output_count,])))
train_dataset = _train_dataset.batch(batch_size,drop_remainder=False)
train_dataset = train_dataset.map(image_preprocess, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.cache(f'{obj_name}_train_cache')
# train_dataset = train_dataset.map(arg_data, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.prefetch(AUTOTUNE).repeat()

# Validation Data
_val_dataset = tf.data.Dataset.from_generator(
            val_datas,
            (tf.float32, tf.uint8),
            (raw_data_shape, tf.TensorShape([output_count,])))
val_dataset = _val_dataset.batch(batch_size,drop_remainder=False).map(
                            image_preprocess, num_parallel_calls=AUTOTUNE
                            ).map(resize,num_parallel_calls=AUTOTUNE
                            ).map(random_crop_frames,num_parallel_calls=AUTOTUNE
                            ).cache(f'{obj_name}_val_cache'
                            ).prefetch(AUTOTUNE).repeat()



str_date = datetime.now().strftime('%Y-%m-%dT%H:%M')
tensorboard_history = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/'+'Video_Conv/' + obj_name+ "_" + str_date,
    profile_batch=0,
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq='epoch')


MODEL_SAVE_FOLDER_PATH = Path(f'model/Video_Conv3D/{obj_name}_{str_date}')
model_path = MODEL_SAVE_FOLDER_PATH/'weights.{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}'
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(model_path),
                    monitor='val_accuracy',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False)


schedule = SGDRScheduler(min_lr=1e-6,
                         max_lr=1e-3,
                         steps_per_epoch=np.ceil(train_count/batch_size),
                         lr_decay=0.8,
                         cycle_length=10,
                         mult_factor=1.0)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = SlowFastNet(input_shape, resnet_size, output_count)
    
model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
    metrics=['accuracy'])
 
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epoch_size,
    steps_per_epoch=np.ceil(train_count/batch_size),
    validation_steps=np.ceil(val_count/batch_size),
    use_multiprocessing=True,
    max_queue_size=300,
    workers=16,
    callbacks=[cb_checkpoint, schedule, tensorboard_history])

tf.saved_model.save(model,f'tmp/{obj_name}')

########################################################################

# loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
# optimizer = tf.keras.optimizers.Adam()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


# def train_step(inp, t):
#     with tf.GradientTape() as tape:
#         prediction = model(inp)
#         loss = loss_function(t, prediction)

#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients,model.trainable_variables))
#     train_loss(loss)

# def train(train_batches):
#     for epoch in range(120):
#         train_loss.reset_states()
#         train_accuracy.reset_states()
#         for batch, (inp, tar) in enumerate(train_batches):

#             train_step(inp, tar)
#             if batch % 10 == 0:
#                 print(
#                 f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
# train(train_dataset)
