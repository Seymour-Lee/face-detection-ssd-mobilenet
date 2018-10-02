# Face-Detection-SSD-MobileNet

## Prerequisites

#### Install TensorFlow Object Detection API  

https://github.com/tensorflow/models/tree/master/research/object_detection

Remember to export the library in PYTHONPATH in your environment.

## Preprocess the dataset

Please run the following scripts:

```shell
python 1_download_data.py

python3 2_data_to_pascal_xml.py

python 3_xml_to_csv.py

python 4_generate_tfrecord.py --images_path=data/tf_wider_train/images --csv_input=data/tf_wider_train/train.csv  --output_path=data/train.record

python 4_generate_tfrecord.py --images_path=data/tf_wider_val/images --csv_input=data/tf_wider_val/val.csv  --output_path=data/val.record

```



## Modify the config file

Read the comments and modify the config information in ssd_mobilenet_v1_face.config



## Train

Just run:

```shell
python models/research/object_detection/train.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_face.config  --train_dir=checkpoints_dir
```



## Export Model

You can export the trained models using this: 

```shell
python models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_face.config --trained_checkpoint_prefix checkpoints_dir/model.ckpt-200 --output_directory output_model/
```

Please modify the name of trained_checkpoint_prefix, like checkpoints_dir/model.ckpt-*number*, where *number* is the num_step in config file



## Eval

You can evaluate the performance of your models using:

```shell
python models/research/object_detection/eval.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_face.config  --checkpoint_dir=checkpoints_dir --eval_dir=eval
```




## Run
Just run:

```shell
python detect_face.py
```








