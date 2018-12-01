
import numpy as np
import scaledconjugategradient as scg
import mlutils as ml  # for draw()
from copy import copy
import sys  # for sys.float_info.epsilon
import pdb

######################################################################
### class NeuralNetwork
######################################################################

class NeuralNetwork:

    def __init__(self, ni,nhs,no):        
        if nhs == 0 or nhs == [0] or nhs is None or nhs == [None]:
            nhs = None
        else:
            try:
                nihs = [ni] + list(nhs)
            except:
                nihs = [ni] + [nhs]
                nhs = [nhs]
        if nhs is not None:
            self.Vs = [(np.random.uniform(-1,1,size=(1+nihs[i],nihs[i+1])) / np.sqrt(nihs[i]))  for i in range(len(nihs)-1)]
            self.W = np.zeros((1+nhs[-1],no))
            # self.W = (np.random.uniform(-1,1,size=(1+nhs[-1],no)) / np.sqrt(nhs[-1]))
        else:
            self.Vs = None
            self.W = np.zeros((1+ni,no))
            # self.W = 0*np.random.uniform(-1,1,size=(1+ni,no)) / np.sqrt(ni)
        self.ni,self.nhs,self.no = ni,nhs,no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.trained = False
        self.reason = None
        self.errorTrace = None
        self.numberOfIterations = None

    def train(self,X,T,nIterations=100,verbose=False,
              weightPrecision=0,errorPrecision=0):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self._standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self._standardizeT(T)

        # Local functions used by scg()

        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            return 0.5 * np.mean((Y - T)**2)

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            delta = (Y - T) / (X.shape[0] * T.shape[1])
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            verbose=verbose,
                            ftracep=True)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        return self

    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        Y = self._unstandardizeT(Y)
        if Z is None:
            return (Y,None) if allOutputs else Y
        else:
            return (Y,Z[1:]) if allOutputs else Y

    def getNumberOfIterations(self):
        return self.numberOfIterations
    
    def getErrorTrace(self):
        return self.errorTrace
        
    def draw(self,inputNames = None, outputNames = None):
        ml.draw(self.Vs+[self.W], inputNames, outputNames)

    def _forward_pass(self,X):
        if self.nhs is None:
            # no hidden units, just linear output layer
            Y = np.dot(X, self.W[1:,:]) + self.W[0:1,:]
            Zs = [X]
        else:
            Zprev = X
            Zs = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Zs.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        return Y, Zs

    def _backward_pass(self,delta,Z):
        if self.nhs is None:
            # no hidden units, just linear output layer
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[0].T, delta)))
            dVs = None
        else:
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
        return dVs,dW

    def _standardizeX(self,X):
        result = (X - self.Xmeans) / self.XstdsFixed
        # result[:,self.Xconstant] = 0.0
        return result
    def _unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def _standardizeT(self,T):
        result = (T - self.Tmeans) / self.TstdsFixed
        # result[:,self.Tconstant] = 0.0
        return result
    def _unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans
   
    def _pack(self,Vs,W):
        if Vs is None:
            return np.array(W.flat)
        else:
            return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self,w):
        if self.nhs is None:
            self.W[:] = w.reshape((self.ni+1, self.no))
        else:
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))

    def __repr__(self):
        str = 'NeuralNetwork({}, {}, {})'.format(self.ni,self.nhs,self.no)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,
                                                                                           self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str
            

######################################################################
### class NeuralNetworkClassifier
######################################################################

def makeIndicatorVars(T):
    """ Assumes argument is N x 1, N samples each being integer class label """
    return (T == np.unique(T)).astype(int)

class NeuralNetworkClassifier(NeuralNetwork):

    def __init__(self,ni,nhs,no):
        #super(NeuralNetworkClassifier,self).__init__(ni,nh,no)
        NeuralNetwork.__init__(self,ni,nhs,no)

    def _multinomialize(self,Y):
        # fix to avoid overflow
        mx = max(0,np.max(Y))
        expY = np.exp(Y-mx)
        # print('mx',mx)
        denom = np.sum(expY,axis=1).reshape((-1,1)) + sys.float_info.epsilon
        Y = expY / denom
        return Y

    def train(self,X,T,
                 nIterations=100,weightPrecision=0,errorPrecision=0,verbose=False):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self._standardizeX(X)

        self.classes, counts = np.unique(T,return_counts=True)
        self.mostCommonClass = self.classes[np.argmax(counts)]  # to break ties

        if self.no != len(self.classes):
            raise ValueError(" In NeuralNetworkClassifier, the number of outputs must equal\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetworkClassifier()." % (self.no, len(self.classes)))
        T = makeIndicatorVars(T)

        # Local functions used by gradientDescent.scg()
        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            Y = self._multinomialize(Y)
            Y[Y==0] = sys.float_info.epsilon
            return -np.mean(T * np.log(Y))

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            Y = self._multinomialize(Y)
            delta = (Y - T) / (X.shape[0] * (T.shape[1]))
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace) - 1
        self.trained = True
        return self
    
    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        Y = self._multinomialize(Y)
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        # If any row has all equal values, then all classes have same probability.
        # Let's return the most common class in these cases
        classProbsEqual = (Y == Y[:,0:1]).all(axis=1)
        if sum(classProbsEqual) > 0:
            classes[classProbsEqual] = self.mostCommonClass
        if Z is None:
            return (classes,Y,None) if allOutputs else classes
        else:
            return (classes,Y,Z[1:]) if allOutputs else classes

######################################################################
### Test code, not run when this file is imported
######################################################################

if __name__== "__main__":
    import matplotlib.pyplot as plt
    plt.ion()  # for use in ipython

    print( '\n------------------------------------------------------------')
    print( "Regression Example: Approximate f(x) = 1.5 + 0.6 x + 0.4 sin(x)")
    # print( '                    Neural net with 1 input, 5 hidden units, 1 output')
    nSamples = 10
    X = np.linspace(0,10,nSamples).reshape((-1,1))
    T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
    T[np.logical_and(X > 2, X < 3)] *= 3
    T[np.logical_and(X > 5, X < 7)] *= 3
    
    nSamples = 100
    Xtest = np.linspace(0,10,nSamples).reshape((-1,1)) + 10.0/nSamples/2
    Ttest = 1.5 + 0.6 * Xtest + 0.8 * np.sin(1.5*Xtest) + np.random.uniform(-2,2,size=(nSamples,1))
    Ttest[np.logical_and(Xtest > 2, Xtest < 3)] *= 3
    Ttest[np.logical_and(Xtest > 5,Xtest < 7)] *= 3

    # # nnet = NeuralNetwork(1,(5,4,3,2),1)
    # # nnet = NeuralNetwork(1,(10,2,10),1)
    # # nnet = NeuralNetwork(1,(5,5),1)
    nnet = NeuralNetwork(1,(3,3,3,3),1)
    
    nnet.train(X,T,errorPrecision=1.e-10,weightPrecision=1.e-10,nIterations=1000)
    print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
    Y = nnet.use(X)
    Ytest,Ztest = nnet.use(Xtest, allOutputs=True)
    print( "Final RMSE: train", np.sqrt(np.mean((Y-T)**2)),"test",np.sqrt(np.mean((Ytest-Ttest)**2)))

    # import time
    # t0 = time.time()
    # for i in range(100000):
    #     Ytest,Ztest = nnet.use(Xtest, allOutputs=True)
    # print( 'total time to make 100000 predictions:',time.time() - t0)
    
    # print( 'Inputs, Targets, Estimated Targets')
    # print( np.hstack((X,T,Y)))

    plt.figure(1)
    plt.clf()
    
    nHLayers = len(nnet.nhs)
    nPlotRows = 3 + nHLayers

    plt.subplot(nPlotRows,2,1)
    plt.plot(nnet.getErrorTrace())
    plt.xlabel('Iterations');
    plt.ylabel('RMSE')
    
    plt.title('Regression Example')
    plt.subplot(nPlotRows,2,3)
    plt.plot(X,T,'o-')
    plt.plot(X,Y,'o-')
    plt.text(8,12, 'Layer {}'.format(nHLayers+1))
    plt.legend(('Train Target','Train NN Output'),loc='lower right',
               prop={'size':9})
    plt.subplot(nPlotRows,2,5)
    plt.plot(Xtest,Ttest,'o-')
    plt.plot(Xtest,Ytest,'o-')
    plt.xlim(0,10)
    plt.text(8,12, 'Layer {}'.format(nHLayers+1))
    plt.legend(('Test Target','Test NN Output'),loc='lower right',
               prop={'size':9})
    colors = ('blue','green','red','black','cyan','orange')
    for i in range(nHLayers):
        layer = nHLayers-i-1
        plt.subplot(nPlotRows,2,i*2+7)
        plt.plot(Xtest,Ztest[layer]) #,color=colors[i])
        plt.xlim(0,10)
        plt.ylim(-1.1,1.1)
        plt.ylabel('Hidden Units')
        plt.text(8,0, 'Layer {}'.format(layer+1))

    plt.subplot(2,2,2)
    nnet.draw(['x'],['sine'])
    plt.draw()

    
    # Now train multiple nets to compare error for different numbers of hidden layers

    if False:  # make True to run multiple network experiment
        
        def experiment(hs,nReps,nIter,X,T,Xtest,Ytest):
            results = []
            for i in range(nReps):
                nnet = NeuralNetwork(1,hs,1)
                nnet.train(X,T,weightPrecision=0,errorPrecision=0,nIterations=nIter)
                # print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
                (Y,Z) = nnet.use(X, allOutputs=True)
                Ytest = nnet.use(Xtest)
                rmsetrain = np.sqrt(np.mean((Y-T)**2))
                rmsetest = np.sqrt(np.mean((Ytest-Ttest)**2))
                results.append([rmsetrain,rmsetest])
            return results

        plt.figure(2)
        plt.clf()
    
        results = []
        # hiddens [ [5]*i for i in range(1,6) ]
        hiddens = [[12], [6,6], [4,4,4], [3,3,3,3], [2,2,2,2,2,2],
                   [24], [12]*2, [8]*3, [6]*4, [4]*6, [3]*8, [2]*12]

        for hs in hiddens:
            r = experiment(hs,30,100,X,T,Xtest,Ttest)
            r = np.array(r)
            means = np.mean(r,axis=0)
            stds = np.std(r,axis=0)
            results.append([hs,means,stds])
            print( hs, means,stds)
        rmseMeans = np.array([x[1].tolist() for x in results])
        plt.clf()
        plt.plot(rmseMeans,'o-')
        ax = plt.gca()
        plt.xticks(range(len(hiddens)),[str(h) for h in hiddens])
        plt.setp(plt.xticks()[1], rotation=30)
        plt.ylabel('Mean RMSE')
        plt.xlabel('Network Architecture')

        
    print( '\n------------------------------------------------------------')
    print( "Classification Example: XOR, approximate f(x1,x2) = x1 xor x2")
    print( '                        Using neural net with 2 inputs, 3 hidden units, 2 outputs')
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    T = np.array([[1],[2],[2],[1]])
    nnet = NeuralNetworkClassifier(2,(4,),2)
    nnet.train(X,T,weightPrecision=1.e-10,errorPrecision=1.e-10,nIterations=100)
    print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
    (classes,y,Z) = nnet.use(X, allOutputs=True)
    

    print( 'X(x1,x2), Target Classses, Predicted Classes')
    print( np.hstack((X,T,classes)))

    print( "Hidden Outputs")
    print( Z)
    
    plt.figure(3)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(np.exp(-nnet.getErrorTrace()))
    plt.xlabel('Iterations');
    plt.ylabel('Likelihood')
    plt.title('Classification Example')
    plt.subplot(2,1,2)
    nnet.draw(['x1','x2'],['xor'])

    
        
