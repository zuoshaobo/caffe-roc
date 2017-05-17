#coding=utf-8
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *



def excute_test(pairlist, test_data):

    result_fold = "result/"
    pair_list = loadmat(pairlist)['lfw_pair']
    test_Intra = pair_list['Inter'][0][0] - 1
    test_Extra = pair_list['Extra'][0][0] - 1



    '''
    pair_list = loadmat(pairlist)
    a = pair_list['pos_pair'] - 1
    b = pair_list['neg_pair']- 1
    test_Intra=[]
    test_Extra=[]
    for i in range(a.shape[1]):
	    test_Intra.append(a[:,i])
    for i in range(b.shape[1]):
	    test_Extra.append(b[:,i])

    test_Intra=np.array(test_Intra)
    test_Extra=np.array(test_Extra)
    '''








    print  test_Intra.shape
    print  test_Extra.shape

    data = np.load(test_data)



    '''
    xmax,xmin = data.max(), data.min()
    data = (data - xmin)/(xmax - xmin)
    data  = data_pre(data)


    pca = PCA_Train(data, result_fold)
    data_pca = pca.transform(data)
    
    clt_pca = joblib.load(result_fold+"pca_model.m")
    data = clt_pca.transform(data)
    data_to_pkl(data, result_fold+"pca_lfw.pkl")

    data = read_pkl(result_fold+"pca_lfw.pkl")
    '''


    #joint bayasian
    with open(result_fold+"A.pkl", "rb") as f:
    	A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
    	G = pickle.load(f)

    dist_Intra = get_ratiosJoint(A,G,test_Intra, data)
    dist_Extra = get_ratiosJoint(A,G,test_Extra, data)



   
    #dist_Intra = get_ratios(test_Intra, data)
    #dist_Extra = get_ratios(test_Extra, data)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label    = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))

    data_to_pkl({"distance": dist_all, "label": label}, result_fold+"result.pkl")


if __name__ == "__main__":

    pairlist = sys.argv[1]
    test_data = sys.argv[2]

    thresfold_start = float(sys.argv[3])
    thresfold_end = float(sys.argv[4])
    thresfold_gap = float(sys.argv[5])

    excute_test(pairlist, test_data)

    excute_performance("result/result.pkl", thresfold_start, thresfold_end, thresfold_gap)
