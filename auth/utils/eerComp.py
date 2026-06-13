'''
@Description: Calculate equal error rate
@Date: 2019-11-27 21:29:36
@LastEditors: lxy
@LastEditTime: 2019-11-28 10:07:10
'''
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize

def calculate_eer(probe, 
                  gallery, 
                  probeLabels, 
                  galleryLabels, 
                  useCosine=1):

    """
    Function to compute EER

    INPUTS:-
    1) probe: Probe embeddings of shape (N,d)
    2) gallery: Gallery embeddings of shape (M,d)
    3) probeLabels: ID labels for probe, shape -> (N,)
    4) galleryLabels: ID labels for gallery, shape -> (M,) 
    5) useCosine: If 1, then cosine distance is used as the distance metric

    OUTPUTS:-
    1) eerVal: Computed EER value
    2) bestThresh: Best value of threshold (i.e., the value at which EER is minimum)
    """

    probe = normalize(probe)
    gallery = normalize(gallery) 

    if(useCosine == 1):
        distVec = distance.cdist(probe,gallery,metric='cosine')
    else:
        distVec = distance.cdist(probe,gallery,metric='euclidean')

    distVec = np.ravel(distVec)
    labelVec = np.ravel(np.expand_dims(probeLabels,-1) == np.expand_dims(galleryLabels,0))

    min_dis = np.min(distVec)
    max_dis = np.max(distVec)

    accept_distances = distVec[labelVec == True]
    reject_distances = distVec[labelVec == False]

    FARs = []
    FRRs = []
    thresholds = []
    errors = []
    for threshold in np.linspace(min_dis, max_dis, num=10000): # threshold linspace
        thresholds.append(threshold)
        FRR = np.sum(accept_distances >= threshold) / accept_distances.shape[0]
        FAR = np.sum(reject_distances < threshold) / reject_distances.shape[0]
        FRRs.append(FRR)
        FARs.append(FAR)
        errors.append(abs(FAR - FRR))
    min_errors_idx = np.argmin(np.asarray(errors))
    eerVal = (FRRs[min_errors_idx] + FARs[min_errors_idx]) / 2
    bestThresh = thresholds[min_errors_idx]

    return eerVal, bestThresh

if __name__ == "__main__":

    probe = np.random.normal(size=(100,32))
    gallery = np.random.normal(size=(100,32))
    
    probeLabel = np.random.randint(0,10,size=(100,))
    galleryLabel = np.random.randint(0,10,size=(100,))

    eerVal, bestThresh = calculate_eer(probe,
                                       gallery,
                                       probeLabel,
                                       galleryLabel,
                                       useCosine=0)
    print(eerVal, bestThresh)