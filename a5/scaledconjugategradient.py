# ScaledConjugateGradient.py
# by Chuck Anderson, for CS545 and CS645
# http://www.cs.colostate.edu/~anderson/cs645
# You may use, but please credit the source.

from copy import copy
import numpy as np
import sys
from math import sqrt, ceil
import time

floatPrecision = sys.float_info.epsilon

######################################################################
# Scaled Conjugate Gradient algorithm from
#  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
#  by Martin F. Moller
#  Neural Networks, vol. 6, pp. 525-533, 1993
#
#  Adapted by Chuck Anderson from the Matlab implementation by Nabney
#   as part of the netlab library.
#
#  Call as   scg()  to see example use.


def scg(x, f, gradf, *fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    evalFunc = params.pop("evalFunc",lambda x: x)
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0) 
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library
  
    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e20			# Upper bound on scale.
    j = 1				# j counts number of iterations.
    
    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None
        
    ### Main optimization loop.
    while j <= nIterations:
        if verbose:
            startTime = time.time()

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if True and kappa < floatPrecision:
                # print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on machine precision"}
            sigma = sigma0/sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        if delta == 0:
            success = False
            fnow = fold
        else:
            alpha = -mu/delta

            ## Calculate the comparison ratio.
            xnew = x + alpha * d
            fnew = f(xnew, *fargs)
            Delta = 2 * (fnew - fold) / (alpha*mu)
            # if np.isnan(Delta):
            #     pdb.set_trace()
            if not np.isnan(Delta) and Delta  >= 0:
                success = True
                nsuccess += 1
                x = xnew
                fnow = fnew
            else:
                success = False
                fnow = fold

        if verbose and j % max(1, ceil(nIterations/10)) == 0:
            print("SCG: Iteration {:d} ObjectiveF={:.5f} Scale={:.3e} Time={:.5f} s/iter".format(j, evalFunc(fnow), beta, (time.time()-startTime)/10.0))
            
        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = evalFunc(fnow)
            
        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 
                        'ftrace':ftrace[:j] if ftracep else None,
                        'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start 
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew
        j += 1

        # if iterationVariable is not None:
        #     iterationVariable.value = j

        ## If we get here, then we haven't terminated in the given number of 
        ## iterations.

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
            'reason':"did not converge"}

######################################################################
### steepest
def steepest(x, f,gradf, *fargs, **params):
    """steepest:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = steepest(firstx, parabola, parabolaGrad, center, S,
                 stepsize=0.01,xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    stepsize= params.pop("stepsize",0.1)
    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision", 1.e-8)  # 1.e-8 is a default value
    fPrecision = params.pop("fPrecision", 1.e-8)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)

    xtracep = True
    ftracep = True
    
    i = 1
    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None
    oldf = f(x,*fargs)
    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = f(x,*fargs)
    else:
        ftrace = None
  
    while i <= nIterations:
        g = gradf(x,*fargs)
        newx = x - stepsize * g
        newf = f(newx,*fargs)
        if (i % (nIterations/10)) == 0:
            print("Steepest: Iteration",i,"Error",evalFunc(newf))
        if xtracep:
            xtrace[i,:] = newx
        if ftracep:
            ftrace[i] = newf
        if np.any(newx == np.nan) or newf == np.nan:
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if np.any(newx==np.inf) or  newf==np.inf:
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if max(abs(newx - x)) < xPrecision:
            return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i],
                    'reason':"limit on x precision"}
        if abs(newf - oldf) < fPrecision:
            return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i],
                    'reason':"limit on f precision"}
        x = newx
        oldf = newf
        i += 1

    return {'x':newx, 'f':newf, 'nIterations':i, 'xtrace':xtrace[:i,:], 'ftrace':ftrace[:i], 'reason':"did not converge"}



if __name__ == "__main__":

    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])

    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)

    print('Stopped after',r['nIterations'],'iterations. Reason for stopping:',r['reason'])
    print('Optimal: point =',r['x'],'f =',r['f'])
