import pickle

## 3rd party packages
import numpy as np

from pca import PCAL1
from processFrames import constructImageFrames

## TODO: create arg parser module to specify image, data file path
INPUT_FILEPATH = "data.pkl"

## TODO: move these to a different file
## Helper functions
def rgb2Grey( rgbImg ):
    """ Given img matrix of sizeMxNx3, return greyscale image matrix of size MxN """
    return np.dot( rgbImg[...,:3], [ 0.2989, 0.5870, 0.1140 ] )

if __name__ == "__main__":
    assert "pkl" == INPUT_FILEPATH.split( '.' )[ -1 ],\
            "Input data file must be in .pkl format"
    with open( INPUT_FILEPATH, "rb" ) as f:
        frameData = pickle.load( f ).astype( float )/255
    nComp = 5
    model = PCAL1( nComp, iteration=10 )
    frameDataNew = model.fitTransform( frameData )

    constructImageFrames( frameData, frameDataNew, 0.1 )