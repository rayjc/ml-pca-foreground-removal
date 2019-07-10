import numpy as np
import processFrames

from scipy.optimize import minimize

class PCAL1:
    """
    PCA model using L1 objective function rather than traditional L2 objective
    function to identify outliers.
    Given:
        k number of components, number of iteration (optional)
        and eps (optional) for multi-quardic approximation for L1 approximation
    Return:
        new X composed by PCA by calling fitTransform()
    """
    def __init__( self, k, iteration=20, eps=0.0001 ):
        self.k = k
        self.iteration = iteration
        self.eps = eps

    def _fit( self, X, imgDim ):
        n,d = X.shape
        k = self.k
        self.mu = np.mean( X, 0 )
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn( n * k )
        w = np.random.randn( k * d )
        with processFrames.createFigureWrapper() as fig:
            for i in range( self.iteration ):
                zResult = minimize( self.zObjFunc, z, args=( w, X, k ),
                                    method="CG", jac=True, options={'maxiter':10} )
                z, fz = zResult.x, zResult.fun
                wResult = minimize( self.wObjFunc, w, args=( z, X, k ),
                                    method="CG", jac=True, options={'maxiter':10} )
                w, fw = wResult.x, wResult.fun
                print('Iteration %d, loss = %.1f, %.1f' % (i, fz, fw))
                processFrames.plotSeparatedImage( X[ 0 ] + self.mu,
                                                    ( z.reshape( n, k ) @ w.reshape(k,d) + self.mu )[ 0 ],
                                                    0.1, fig, imgDim )

        self.W = w.reshape(k,d)

    def _compress( self, X ):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # W may not be orthogonal so we need to solve for Z one last time...
        z = np.zeros( n * k )
        zResult = minimize( self.zObjFunc, z, args=( self.W.flatten(), X, k ),
                            method="CG", jac=True, options={'maxiter':100} )
        z, fz = zResult.x, zResult.fun
        Z = z.reshape( n, k )
        return Z

    def _expand( self, Z ):
        X = Z @ self.W + self.mu
        return X

    def fitTransform( self, X, imgDim ):
        self._fit( X, imgDim )
        return self._expand( self._compress( X ) )

    def zObjFunc( self, z, w, X, k ):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum( abs( R ) )
        R[:] /= np.sqrt( R[:]**2 + self.eps )
        g = np.dot( R, W.transpose() )
        #print( "loss (fixed w): ", f )
        return f, g.flatten()

    def wObjFunc( self, w, z, X, k ):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum( abs( R ) )
        R[:] /= np.sqrt( R[:]**2 + self.eps )
        g = np.dot( Z.transpose(), R )
        #print( "loss (fixed z): ", f )
        return f, g.flatten()
