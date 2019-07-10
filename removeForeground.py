import pickle

## 3rd party packages
import numpy as np
import miscHelper

from pca import PCAL1
from processFrames import constructImageFrames

## TODO: move these to a different file
## Helper functions
def rgb2Grey( rgbImg ):
    """ Given img matrix of sizeMxNx3, return greyscale image matrix of size MxN """
    return np.dot( rgbImg[...,:3], [ 0.2989, 0.5870, 0.1140 ] )

def main():

    cli = miscHelper.CliConfig()
    if cli.config[ "subParser" ] == "train":
        with open( cli.config[ "data" ], "rb" ) as f:
            frameData = pickle.load( f ).astype( float )/255
        assert cli.config[ "size" ][ 0 ] * cli.config[ "size" ][ 1 ] == frameData.shape[ 1 ],\
                """
                The specified image resolution does not match data in .pkl file.
                The default size is {} by {}. Otherwise, you may choose to specify -s option.
                """.format( miscHelper.imgDim[ 0 ], miscHelper.imgDim[ 1 ] )
        
        ## TODO: add interactive check by displaying one reshaped image frame
        nComp = 5
        model = PCAL1( nComp, iteration=10 )
        frameDataNew = model.fitTransform( frameData, cli.config[ "size" ] )

        constructImageFrames( frameData, frameDataNew, 0.1, cli.config[ "size" ], cli.config[ "output" ] )


if __name__ == "__main__":
    main()