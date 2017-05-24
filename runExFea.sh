#!/bin/sh

name=whatyouwant

trainNet=/home/pub/Work/face_verification_experiment/proto/LightenedCNN_B_deploy.prototxt
caffemodel=/home/pub/Work/BWN-XNOR-caffe/o_sony__iter_93756.caffemodel
#trainNet=/home/pub/Work/face_verification_experiment/proto/Binary_LightenedCNN_B_deploy.prototxt.laskinnerproduct
#caffemodel=/home/pub/Work/BWN-XNOR-caffe/lightcnn/binarylastinnerproduct_l_sony__iter_good.caffemodel
#trainNet=/media/pub/AE222BE6222BB1EF/Users/zuoshaobo/Desktop/stream/FaceRecognition_sdk/face_recog.prototxt
#caffemodel=/media/pub/AE222BE6222BB1EF/Users/zuoshaobo/Desktop/stream/FaceRecognition_sdk/face_recog.caffemodel
meanFile=rgb_50x50_mean.npy
filelist=lfw_list2.txt
#dimension=4096
#imagesize=224
dimension=256
imagesize=128
outputPath=$name



# extract the deep feature from data by caffmodels and trainNet that you are using

python extraction.py $trainNet $caffemodel model/image_mean/$meanFile $filelist $dimension $imagesize feature/${name}

echo "extractin done.."


