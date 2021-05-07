#!/bin/bash

# Download pretrained model

#works on opencv http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt

here=`pwd`
config="ssd_mobilenet_v1_fpn.config"
pretrained_models="models/pretrained_models"
tfmodels_dir="tfdetection/"

rm -rf ${pretrained_models}
mkdir -p ${pretrained_models}
cd ${pretrained_models}
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
gunzip -c ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz | tar xopf -
rm ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
mv ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 ssd_mobilenet_v1_fpn
cd $here

# Training

cd ${tfmodels_dir}
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/object_detection:`pwd`/slim
export LC_ALL="en_US.UTF-8"
cd $here


PIPELINE_CONFIG_PATH="$here/$config"
MODEL_DIR="/tmp/training/ssd_mobilenet_v1_fpn"
rm -rf ${MODEL_DIR}
mkdir -p ${MODEL_DIR}
NUM_TRAIN_STEPS=100000
NUM_EVAL_STEPS=500
tensorboard --logdir=${MODEL_DIR} &


#python ${tfmodels_dir}object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --num_train_steps=${NUM_TRAIN_STEPS} \
#    --num_eval_steps=${NUM_EVAL_STEPS} \
#    --alsologtostderr

python ${tfmodels_dir}/object_detection/legacy/train.py --logtostderr --train_dir=/tmp/training/ --pipeline_config_path=$config

#Conversion

rm -rf bin-graph
mkdir bin-graph
rm -rf tmp-bin-graph
mkdir tmp-bin-graph

i=1

for f in /tmp/training/model.ckpt-*.index; do
    model=${f/.index/}
    IFS='-' read -r -a array <<< "$model"
    idx=${array[1]}

    echo "Converting " $model "to model_"${idx}".pb"


    python3 -m object_detection.export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path $config \
    --trained_checkpoint_prefix $model \
    --output_directory tmp-bin-graph

    mv tmp-bin-graph/frozen_inference_graph.pb bin-graph/model_${idx}.pb
    rm -rf tmp-bin-graph/*

    if [[ -d /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer ]]; then
        echo "make openvino models"
        here=`pwd`

        cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer

        python3 mo_tf.py --input_model=${here}/bin-graph/model_${idx}.pb \
        --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json \
        --tensorflow_object_detection_api_pipeline_config ${here}/${config} \
        --output_dir $here/openvino/ --data_type FP16

        python3 mo_tf.py --input_model=${here}/bin-graph/model_${idx}.pb \
        --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json \
        --tensorflow_object_detection_api_pipeline_config ${here}/${config} \
        --output_dir $here/openvino-cpu/ --data_type FP32

        cd ${here}
    fi
    ((i++))
done
