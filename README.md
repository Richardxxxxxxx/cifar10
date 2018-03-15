# cifar10

This is to using inception to classify cifar10 dataset

```shell

#recover image from cifar10 batch file, edit three valiables
#base_path = base path where to store data_batch* and test_batch
#train_dir = path to store recover training data
#validate_dir = path to store recover testing data

python convert.py



#process your converted cifar10 data
python build_image_data.py \
--train_directory='/Users/apple/Desktop/cifar10data/cifar-10-batches-py/train' \
--validation_directory='/Users/apple/Desktop/cifar10data/cifar-10-batches-py/validate' \
--output_directory='/Users/apple/Desktop/cifar10data/processed-data' \
--labels_file='/Users/apple/Desktop/cifar10data/label' \
--train_shards=8 \
--validation_shards=1 \
--num_threads=1



#running ps node
imagenet_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts='127.0.0.1:2222' \
--worker_hosts='127.0.0.1:2223'

#running worker node
python imagenet_distributed_train.py \
--train_dir='/Users/apple/Desktop/cifar10data/model' \
--data_dir='/Users/apple/Desktop/cifar10data/processed-data' \
--batch_size=128 \
--job_name='worker' \
--max_steps=1000 \
--initial_learning_rate=0.1 \
--num_epochs_per_decay=350 \
--learning_rate_decay_factor=0.1 \
--task_id=0 \
--ps_hosts='127.0.0.1:2222' \
--worker_hosts='127.0.0.1:2223'


```

for more details please refer to Original Source:https://github.com/tensorflow/models/tree/master/research/inception
