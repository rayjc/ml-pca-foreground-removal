import pickle

## 3rd party packages
import numpy as np
import miscHelper

from pca import PCAL1
from processFrames import constructImageFrames, createPklFile

def main():

    cli = miscHelper.CliConfig()

    if cli.config[ "subParser" ] in ( "preprocess", "all" ):
        pklFileName = createPklFile( cli.config[ "size" ], cli.config[ "input" ],
                        cli.config[ "output" ] )
        print( "Preprocess completed: image data has been compiled into ", pklFileName )

    if cli.config[ "subParser" ] in ( "train", "all" ):
        if cli.config[ "subParser" ] == "train":
            ## Load pre-existing .pkl file
            with open( cli.config[ "data" ], "rb" ) as f:
                frameData = pickle.load( f ).astype( float )/255
        else:
            ## Load .pkl file created during preprocess stage of this session
            with open( pklFileName, "rb" ) as f:
                frameData = pickle.load( f ).astype( float )/255
        assert cli.config[ "size" ][ 0 ] * cli.config[ "size" ][ 1 ] == frameData.shape[ 1 ],\
                """
                The specified image resolution does not match data in .pkl file.
                The default size is {} by {}. Otherwise, you may choose to specify -s option.
                """.format( miscHelper.imgDim[ 0 ], miscHelper.imgDim[ 1 ] )
        
        ## TODO: add interactive check by displaying one reshaped image frame
        model = PCAL1( cli.config[ "component" ], iteration=cli.config[ "iteration" ] )
        frameDataNew = model.fitTransform( frameData, cli.config[ "size" ] )

        constructImageFrames( frameData, frameDataNew, 0.1, cli.config[ "size" ], cli.config[ "output" ] )


if __name__ == "__main__":
    main()