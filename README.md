# face-detection-ssd-mobilenet
face-detection-ssd-mobilenet-tensorflow

Please install Tensorflow Object Detection API first:
https://github.com/tensorflow/models/tree/master/research/object_detection
remember to export PYTHONPATH in .profile like:
export PYTHONPATH=$PYTHONPATH:/Users/miaozou/Documents/projects/models/research:/Users/miaozou/Documents/projects/models/research/slim

## Prepare
python 1_download_data.py

python3 2_data_to_pascal_xml.py

python 3_xml_to_csv.py

python 4_generate_tfrecord.py --images_path=data/tf_wider_train/images --csv_input=data/tf_wider_train/train.csv  --output_path=data/train.record

python 4_generate_tfrecord.py --images_path=data/tf_wider_val/images --csv_input=data/tf_wider_val/val.csv  --output_path=data/val.record

## Modify
Read the comments and modify the config information in ssd_mobilenet_v1_face.config

## Train
python /Users/miaozou/Documents/projects/models/research/object_detection/train.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_face.config  --train_dir=model_output



## Export Model
python /Users/miaozou/Documents/projects/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_face.config --trained_checkpoint_prefix model_output/model.ckpt-20 --output_directory model/


## Eval
python /Users/miaozou/Documents/projects/models/research/object_detection/eval.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_face.config  --checkpoint_dir=model_output --eval_dir=eval


## Run
python detect_face.py