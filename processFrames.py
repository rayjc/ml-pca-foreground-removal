import glob
import os
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import miscHelper
import numpy as np
import skimage.io, skimage.util, skimage.transform

from contextlib import contextmanager

defaultInputDir = "input"
defaultOutputDir = "output"
bgDir = "background"
fgDir = "foreground"
processedDir = "input-preprocessed"

def constructImageFrames( X, XNew, threshold, imgDim, outputName ):
    ## create directories to store the images
    if not os.path.exists( defaultOutputDir ):
        os.mkdir( defaultOutputDir )
        os.mkdir( os.path.join( defaultOutputDir, bgDir ) )
        os.mkdir( os.path.join( defaultOutputDir, fgDir ) )
    else:
        if not os.path.isdir( os.path.join( defaultOutputDir, fgDir ) ):
            os.mkdir( os.path.join( defaultOutputDir, fgDir ) )
        if not os.path.isdir( os.path.join( defaultOutputDir, bgDir ) ):
            os.mkdir( os.path.join( defaultOutputDir, bgDir ) )
    assert( X.shape == XNew.shape )
    nRow, nCol = XNew.shape
    warnings.filterwarnings(action="ignore", category=UserWarning, message=r".*is a low contrast image" )
    warnings.filterwarnings(action="ignore", category=UserWarning, message=r".*Possible precision loss" )
    for row in range( nRow ):
        skimage.io.imsave( os.path.join( defaultOutputDir, bgDir, "{}_bg_{}.jpg".format( outputName, row + 1 ) ),
                            skimage.util.img_as_ubyte( XNew[ row ].reshape( imgDim ) ) )
        fgBool = np.logical_not( np.isclose( X[ row ], XNew[ row ], atol=threshold ) )
        skimage.io.imsave( os.path.join( defaultOutputDir, fgDir, "{}_fg_{}.jpg".format( outputName, row + 1 ) ),
                            skimage.util.img_as_ubyte( np.where( fgBool, X[ row ], 0.0 ).reshape( imgDim ) ) )

@contextmanager
def createFigureWrapper():
    maxWidth, maxHeight = getScreenRes()
    try:
        yield plt.figure( figsize=( maxWidth/200.0, maxHeight/200.0 ) )
    finally:
        plt.close()

def createPklFile( imgDim=tuple( miscHelper.imgDim ), inputDir=defaultInputDir, outputName="unnamed" ):
    """
    Given a input directory of image frames, rescale and convert these images to .jpg
    then create .pkl to this matrix data.
    """
    ## convert any image not in .jpg format to .jpg
    imageFilePaths = glob.glob( os.path.join( defaultInputDir, "*" ) )
    for imagePath in imageFilePaths:
        if ".jpg" not in imagePath:
            img = skimage.io.imread( imagePath, as_gray=True )
            if not os.path.exists( processedDir ):
                os.mkdir( processedDir )
            newImgFilePath = os.path.join( processedDir, 
                                            imagePath.split( os.sep )[ -1 ].split( '.' )[ 0 ] )\
                                            + ".jpg"
            ## save the image as .jpg to compress image data
            ## there should be a way to convert/compress data directly...
            skimage.io.imsave( newImgFilePath,
                                skimage.util.img_as_ubyte( img ) )
            print( "Converted {} to {}".format( imagePath, newImgFilePath ) )
            # os.remove( imagePath )
        else:
            shutil.copy( imagePath, processedDir )

    imageFilePaths = glob.glob( os.path.join( processedDir, "*" ) )
    imgData = None
    ## Construct a NxD matrix of N frames(samples) and D pixels(features)
    for imagePath in imageFilePaths:
        assert ".jpg" in imagePath
        ## Read and rescale each image
        img = skimage.io.imread( imagePath, as_gray=True )
        rescaledImg = skimage.util.img_as_ubyte( skimage.transform.resize( img, imgDim ) )
        flatImg = rescaledImg.reshape( ( 1, imgDim[ 0 ] * imgDim[ 1 ] ) )
        if isinstance( imgData, np.ndarray ):
            imgData = np.concatenate( ( imgData, flatImg ), axis=0 )
        else:
            imgData = flatImg
        ## Update the image file
        # os.remove( imagePath )
        skimage.io.imsave( imagePath, rescaledImg )
        print( "Updated ", imagePath )
    
    ## Pickle and dump the constructed matrix data set
    dataFileName = "{}.pkl".format( outputName )
    if os.path.exists( dataFileName ):
        os.remove( dataFileName )
    with open( dataFileName, "wb" ) as f:
        pickle.dump( imgData, f )
    return dataFileName

def getScreenRes():
    window = plt.get_current_fig_manager().window
    maxWidth, maxHeight = window.wm_maxsize()
    plt.close()
    return maxWidth, maxHeight

def plotSeparatedImage( frame, background, threshold, figure, imgDim ):
    plt.clf()
    numSubplots = 3
    ax = [ figure.add_subplot( 1, numSubplots, index + 1 ) for index in range( numSubplots ) ]
    ax[ 0 ].set_title( "Original Frame" )
    ax[ 0 ].imshow(frame.reshape( imgDim ), cmap="gray" )
    ax[ 1 ].set_title( "Background" )
    ax[ 1 ].imshow( background.reshape( imgDim ), cmap="gray" )
    fgBool = np.logical_not( np.isclose( frame, background, atol=threshold ) )
    ax[ 2 ].set_title( "Foreground" )
    ax[ 2 ].imshow( np.where( fgBool, frame, 0.0 ).reshape( imgDim ), cmap="gray" )
    plt.show( block=False )
    plt.pause( 0.001 )


if __name__ == "__main__":
    createPklFile()