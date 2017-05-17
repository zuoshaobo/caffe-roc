#!/bin/sh

#name=1
name=whatyouwant

pairlist=filelist/lfw_pair.mat
test_data=feature/${name}.npy
thres_s=-106.9
thres_e=16.6
thres_g=0.01

# bayesain
#thres_s=-106.9
#thres_e=16.6
#thres_g=0.01

# run the ROC results

python testROC/test_lfw.py $pairlist $test_data $thres_s $thres_e $thres_g

echo "done"

