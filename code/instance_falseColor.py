
import numpy as np


n_classes = 21 # for PASCAL
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def get_falseColor(classId, instId, instNum):
    color = np.dot(label_colours[classId], (instId/instNum))
    return color


def enc_falseColorInstanceImage(label, classIds):
    # assgin false color
    outI = np.zeros((label.shape[0], label.shape[1], 3))
    pre_cls     = -1
    pre_instId = 0
    for k in range(len(classIds)):
        clsId = classIds[k]

        if clsId == pre_cls:
            instId = pre_instId +1
        else:
            instId = 1

        color = get_falseColor(clsId, instId, 10)
        logit = label==(k+1)
        outI  = outI + (logit[..., np.newaxis] * color[np.newaxis, ...])

        pre_instId = instId
        pre_cls    = clsId

    return outI

# to do:: decoder false color
'''
def dec_falseColorInstanceImage(colorI):
    ddd

'''

def simpleInstanceInference(predProb, classIds):
    # find max segment
    label =  np.argmax(predProb, axis = 0) + 1
    proImg = np.max(predProb, axis = 0)
    label = label*(proImg > 0.01)

    return enc_falseColorInstanceImage(label, classIds)





