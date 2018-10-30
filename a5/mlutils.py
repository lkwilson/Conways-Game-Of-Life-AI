import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Machine Learning Utilities. 
#
#  confusionMatrix
#  printConfusionMatrix
#  trainValidateTestKFoldsClassification
#  partition
#  partitionsKFolds
#  draw (a neural network)
#  matrixAsSquares
######################################################################

def trainValidateTestKFoldsClassification(trainf,evaluatef,X,T,parameterSets,nFolds,shuffle=False,verbose=False):
    if nFolds < 3:
        raise ValueError('ERROR: trainValidateTestKFoldsClassification requires nFolds >= 3.')
    # Collect row indices for each class
    classes = np.unique(T)
    K = len(classes)
    rowIndicesByClass = []
    for c in classes:
        rowsThisClass = np.where(T == c)[0]
        if shuffle:
            np.random.shuffle(rowsThisClass)
        rowIndicesByClass.append(rowsThisClass)
    # Collect start and stop indices for the folds, within each class
    startsStops = []
    if verbose:
        print('  In each of',nFolds,'folds, Class-Counts ',end="")
    for k,rowIndicesThisClass in enumerate(rowIndicesByClass):
        nSamples = len(rowIndicesThisClass)
        nEach = int(nSamples / nFolds)
        if verbose:
            print('{}-{},'.format(classes[k],nEach),end=" ") #'samples in each of',nFolds,'folds.')
        if nEach == 0:
            raise ValueError("trainValidateTestKFoldsClassification: Number of samples in each fold for class {} is 0.".format(k))
        startsThisClass = np.arange(0,nEach*nFolds,nEach)
        if k < K-1: #last class
            stopsThisClass = startsThisClass + nEach
        else:
            stopsThisClass = startsThisClass + nSamples #Each
        startsStops.append(list(zip(startsThisClass,stopsThisClass)))
    print()

    results = []
    for testFold in range(nFolds):
        # Leaving the testFold out, for each validate fold, train on remaining
        # folds and evaluate on validate fold. 
        bestParms = None
        bestValidationAccuracy = 0
        for parms in parameterSets:
            validateAccuracySum = 0
            for validateFold in range(nFolds):
                if testFold == validateFold:
                    continue
                trainFolds = np.setdiff1d(range(nFolds), [testFold,validateFold])
                rows = []
                for tf in trainFolds:
                    for k in range(K):
                        a,b = startsStops[k][tf]                
                        rows += rowIndicesByClass[k][a:b].tolist()
                Xtrain = X[rows,:]
                Ttrain = T[rows,:]
                # Construct Xvalidate and Tvalidate
                rows = []
                for k in range(K):
                    a,b = startsStops[k][validateFold]
                    rows += rowIndicesByClass[k][a:b].tolist()
                Xvalidate = X[rows,:]
                Tvalidate = T[rows,:]

                model = trainf(Xtrain,Ttrain,parms)
                validateAccuracy = evaluatef(model,Xvalidate,Tvalidate)
                validateAccuracySum += validateAccuracy
            validateAccuracy = validateAccuracySum / (nFolds-1)
            if bestParms is None or validateAccuracy > bestValidationAccuracy:
                bestParms = parms
                bestValidationAccuracy = validateAccuracy
        rows = []
        for k in range(K):
            a,b = startsStops[k][testFold]
            rows += rowIndicesByClass[k][a:b].tolist()
        Xtest = X[rows,:]
        Ttest = T[rows,:]

        newXtrain = np.vstack((Xtrain,Xvalidate))
        newTtrain = np.vstack((Ttrain,Tvalidate))
        model = trainf(newXtrain,newTtrain,bestParms)
        trainAccuracy = evaluatef(model,newXtrain,newTtrain)
        testAccuracy = evaluatef(model,Xtest,Ttest)

        resultThisTestFold = [bestParms, trainAccuracy,
                              bestValidationAccuracy, testAccuracy]
        results.append(resultThisTestFold)
        if verbose:
            print(resultThisTestFold)
    return results

def percentCorrect(predictedClasses, TrueClasses):
    return np.sum(TrueClasses == predictedClasses) / float(TrueClasses.shape[0]) * 100

######################################################################

def confusionMatrix(actual,predicted,classes,probabilities=None,probabilityThreshold=None):
    nc = len(classes)
    if probabilities is not None:
        predictedClassIndices = np.zeros(predicted.shape,dtype=np.int)
        for i,cl in enumerate(classes):
            predictedClassIndices[predicted == cl] = i
        probabilities = probabilities[np.arange(probabilities.shape[0]), predictedClassIndices.squeeze()]
    confmat = np.zeros((nc,nc+2)) # for samples above threshold this class and samples this class
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        if probabilities is None:
            keep = trues
            predictedThisClassAboveThreshold = predictedThisClass
        else:
            keep = probabilities[trues] >= probabilityThreshold
            predictedThisClassAboveThreshold = predictedThisClass[keep]
        # print 'confusionMatrix: sum(trues) is ', np.sum(trues),'for classes[ri]',classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
        confmat[ri,nc] = np.sum(keep)
        confmat[ri,nc+1] = np.sum(trues)
    printConfusionMatrix(confmat,classes)
    return confmat

def printConfusionMatrix(confmat,classes):
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('%s' % '------'*len(classes))
    for i,t in enumerate(classes):
        print('%2d |' % (t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('%5.1f' % (100*confmat[i,i1]), end='')
        print('   (%d / %d)' % (int(confmat[i,len(classes)]), int(confmat[i,len(classes)+1])))

######################################################################

def partition(X,T,fractions,classification=False):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),classification=True)
      X is nSamples x nFeatures.
      fractions can have just two values, for partitioning into train and test only
      If classification=True, T is target class as integer. Data partitioned
        according to class proportions.
        """
    trainFraction = fractions[0]
    if len(fractions) == 2:
        # Skip the validation step
        validateFraction = 0
        testFraction = fractions[1]
    else:
        validateFraction = fractions[1]
        testFraction = fractions[2]
        
    rowIndices = np.arange(X.shape[0])
    np.random.shuffle(rowIndices)
    
    if classification != 1:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        Xtrain = X[rowIndices[:nTrain],:]
        Ttrain = T[rowIndices[:nTrain],:]
        if nValidate > 0:
            Xvalidate = X[rowIndices[nTrain:nTrain+nValidate],:]
            Tvalidate = T[rowIndices[nTrain:nTrain:nValidate],:]
        Xtest = X[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        Ttest = T[rowIndices[nTrain+nValidate:nTrain+nValidate+nTest],:]
        
    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        trainIndices = []
        validateIndices = []
        testIndices = []
        for c in classes:
            # row indices for class c
            cRows = np.where(T[rowIndices,:] == c)[0]
            # collect row indices for class c for each partition
            n = len(cRows)
            nTrain = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate
            trainIndices += rowIndices[cRows[:nTrain]].tolist()
            if nValidate > 0:
                validateIndices += rowIndices[cRows[nTrain:nTrain+nValidate]].tolist()
            testIndices += rowIndices[cRows[nTrain+nValidate:nTrain+nValidate+nTest]].tolist()
        Xtrain = X[trainIndices,:]
        Ttrain = T[trainIndices,:]
        if nValidate > 0:
            Xvalidate = X[validateIndices,:]
            Tvalidate = T[validateIndices,:]
        Xtest = X[testIndices,:]
        Ttest = T[testIndices,:]
    if nValidate > 0:
        return Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
    else:
        return Xtrain,Ttrain,Xtest,Ttest

######################################################################

def partitionsKFolds(X,T,K,validation=True,shuffle=True,classification=True):
    '''Returns Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
      or
       Xtrain,Ttrain,Xtest,Ttest if validation is False
    Build dictionary keyed by class label. Each entry contains rowIndices and start and stop
    indices into rowIndices for each of K folds'''
    global folds
    if not classification:
        print('Not implemented yet.')
        return
    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(rowIndices)
    folds = {}
    classes = np.unique(T)
    for c in classes:
        classIndices = rowIndices[np.where(T[rowIndices,:] == c)[0]]
        nInClass = len(classIndices)
        nEach = int(nInClass / K)
        starts = np.arange(0,nEach*K,nEach)
        stops = starts + nEach
        stops[-1] = nInClass
        # startsStops = np.vstack((rowIndices[starts],rowIndices[stops])).T
        folds[c] = [classIndices, starts, stops]

    for testFold in range(K):
        if validation:
            for validateFold in range(K):
                if testFold == validateFold:
                    continue
                trainFolds = np.setdiff1d(range(K), [testFold,validateFold])
                rows = rowsInFold(folds,testFold)
                Xtest = X[rows,:]
                Ttest = T[rows,:]
                rows = rowsInFold(folds,validateFold)
                Xvalidate = X[rows,:]
                Tvalidate = T[rows,:]
                rows = rowsInFolds(folds,trainFolds)
                Xtrain = X[rows,:]
                Ttrain = T[rows,:]
                yield Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
        else:
            # No validation set
            trainFolds = np.setdiff1d(range(K), [testFold])
            rows = rowsInFold(folds,testFold)
            Xtest = X[rows,:]
            Ttest = T[rows,:]
            rows = rowsInFolds(folds,trainFolds)
            Xtrain = X[rows,:]
            Ttrain = T[rows,:]
            yield Xtrain,Ttrain,Xtest,Ttest
            

def rowsInFold(folds,k):
    allRows = []
    for c,rows in folds.items():
        classRows, starts, stops = rows
        allRows += classRows[starts[k]:stops[k]].tolist()
    return allRows

def rowsInFolds(folds,ks):
    allRows = []
    for k in ks:
        allRows += rowsInFold(folds,k)
    return allRows

######################################################################

######################################################################
# Associated with  neuralnetworks.py
# Draw a neural network with weights in each layer as a matrix
######################################################################


def draw(W, inputNames = None, outputNames = None, gray = False):

    def isOdd(x):
        return x % 2 != 0

    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len,inputNames))/4.0 if inputNames else 1
    ylim = 0
    
    for li in range(nLayers):
        ni,no = W[li].shape  #no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni,no = W[nLayers-1].shape  #no means number outputs this layer
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len,outputNames))/4.0)


    maxw = max([np.max(np.abs(w)) for w in W])
    largestDimNWeights = max( sum([w.shape[i%2] for i,w in enumerate(W)]),
                              sum([w.shape[1-i%2] for i,w in enumerate(W)]))

    # largestDim = max(xlim,ylim)
    # scaleW = 1000000 /  largestDimNWeights / maxw
    # print(largestDim,largestDimNWeights,maxw,scaleW)

    ax = plt.gca()

    x0 = 1
    y0 = 0 # to allow for constant input to first layer
    # First Layer
    if inputNames:
        # addx = max(map(len,inputNames))*0.1
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0-len(n)*0.2, y, n)
            x0 = max([1,max(map(len,inputNames))/4.0])

    for li in range(nLayers):
        Wi = W[li]
        scaleW = 500 /  largestDimNWeights / np.max(np.abs(Wi))
        ni,no = Wi.shape
        if not isOdd(li):
            # Odd layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1')
            for i in range(ni):
                ax.plot((x0,x0+no-0.5), (y0+i+0.5, y0+i+0.5),color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0+1+i-0.5, x0+1+i-0.5), (y0, y0+ni+1),color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            ax.scatter(xs,ys,marker='v',s=100,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(no)+ x0+0.5
            ys = np.arange(ni)+ y0 + 0.5
            aWi = (np.abs(Wi) * scaleW)**2
            coords = np.meshgrid(xs,ys)
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            print(scaleW,np.min(aWi),np.max(aWi))
            ax.scatter(coords[0],coords[1],marker='s',s=aWi,c=colors)
            y0 += ni + 1
            x0 += -1 ## shift for next layer's constant input
        else:
            # Even layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1')
            # input lines
            for i in range(ni):
                ax.plot((x0+i+0.5,  x0+i+0.5), (y0,y0+no-0.5),color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0, x0+ni+1), (y0+i+0.5, y0+i+0.5),color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5]*no)
            ys = y0 + 0.5 + np.arange(no)
            ax.scatter(xs,ys,marker='>',s=100,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs,ys)
            aWi = (np.abs(Wi) * scaleW)**2
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi,c=colors)
            x0 += ni + 1
            y0 -= 1 ##shift to allow for next layer's constant input

    # Last layer output labels 
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n)
    ax.axis([0,xlim, ylim,0])
    ax.axis('off')


def matrixAsSquares(W, maxSize = 200, color = False):

    if color:
        facecolors = np.array(["red","green"])[(W.flat >= 0)+0]
        edgecolors = np.array(["red","green"])[(W.flat >= 0)+0]
    else:
        facecolors = np.array(["black","none"])[(W.flat >= 0)+0]
        edgecolors = np.array(["black","black"])[(W.flat >= 0)+0]

    xs = np.arange(W.shape[1]) #+ x0+0.5
    ys = np.arange(W.shape[0]) #+ y0 + 0.5
    maxw = np.max(np.abs(W))
    aWi = (np.abs(W)/maxw) * maxSize
    coords = np.meshgrid(xs,ys)
    #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
    ax = plt.gca()
    # pdb.set_trace()
    for x, y, width, fc,ec  in zip(coords[0].flat,coords[1].flat, aWi.flat, facecolors, edgecolors):
        ax.add_patch(pltpatch.Rectangle((x+maxSize/2-width/2, y+maxSize/2)-width/2, width,width,facecolor=fc,edgecolor=ec))
    # plt.scatter(coords[0],coords[1],marker='s',s=aWi,
    #            facecolor=facecolors,edgecolor=edgecolors)
    plt.xlim(-.5,W.shape[1]+0.5)
    plt.ylim(-.5,W.shape[0]-0.5)
    plt.axis('tight')

if __name__ == '__main__':
    plt.ion()

    plt.figure(1)
    plt.clf()

    plt.subplot(2,1,1)
    w = np.arange(-100,100).reshape(20,10)
    matrixAsSquares(w,200,color=True)

    plt.subplot(2,1,2)
    w = np.arange(-100,100).reshape(20,10)
    matrixAsSquares(w,100,color=False)

