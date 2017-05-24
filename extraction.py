import sys
import numpy as np
import copy
caffe_root = '/home/pub/Work/BWN-XNOR-caffe'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import skimage.io


def readDeepNet(trainNet_path, caffemodel_path, proj_root):
    net = caffe.Classifier(trainNet_path, caffemodel_path)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_channel_swap('data',(2,1,0))

    return (net, transformer)


def extract_feature(net, transformer, filelist, dimension, imageSize):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    file = open(filelist)

    nan = np.empty(shape=[0, dimension])
    _label = np.empty(shape=[0, 1])
    looptimes = 0
    while 1:
        line = file.readline()
        if not line:
            break
        pass
        
        spaceIndex = line.find(" ")
        imagePath = line[0:spaceIndex]
        thisLabel = int(line[spaceIndex + 1:len(line)])
	imagePath='/home/pub/samba/lfw_aligned/'+imagePath
	print  imagePath,thisLabel

        #img = caffe.io.load_image(imagePath, color = True)
	#'''
        img = caffe.io.load_image(imagePath, color = False)
        if  img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]
	#'''
	net.blobs['data'].data[0]=transformer.preprocess('data',img)

        out = net.forward()
	#print tmp.data[0]
        feat = net.blobs['eltwise_fc1'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model
        #feat = net.blobs['fc7'].data[0] # 'fc160' is the layer name in caffemodel, edit this arguments base on your own model
	#print feat.size
	#exit(0)
	fea=copy.deepcopy(feat)

        nan = np.vstack((nan, feat))
        _label = np.vstack((_label, thisLabel))
    return (nan, _label)


def print_help():
    print "argv: [1]trainNet path; [2]caffemodel path; [3]meanFile path \n"
    print "[4]filelist path; [5]dimension of feature vector; [6]imageSize; [7]output path\n"
    exit


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print_help()
    else:
        trainNet_path = sys.argv[1]
        caffemodel_path = sys.argv[2]
        meanFile_path = sys.argv[3]
        filelist = sys.argv[4]
        dimension = int(sys.argv[5])
        imageSize = int(sys.argv[6])
        (net, transformer) = readDeepNet(trainNet_path, caffemodel_path, meanFile_path)
        (feature, label) = extract_feature(net, transformer, filelist, dimension, imageSize)
        outputFileName = sys.argv[7]

        print(feature.shape)
        np.save(outputFileName, feature)
	print "save finish"
	print "save finish"

