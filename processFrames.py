import glob
import os
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage.io, skimage.util, skimage.transform

from contextlib import contextmanager

imgDim = ( 120, 160 )       ## ( 240, 426 ) for 240p on Youtube
inputDir = "input"
outputDir = "output"
bgDir = "background"
fgDir = "foreground"
processedDir = "input-preprocessed"
outputDataFile = "data.pkl"

def constructImageFrames( X, XNew, threshold ):
    ## create directories to store the images
    if not os.path.exists( outputDir ):
        os.mkdir( outputDir )
        os.mkdir( os.path.join( outputDir, bgDir ) )
        os.mkdir( os.path.join( outputDir, fgDir ) )
    else:
        if not os.path.isdir( os.path.join( outputDir, fgDir ) ):
            os.mkdir( os.path.join( outputDir, fgDir ) )
        if not os.path.isdir( os.path.join( outputDir, bgDir ) ):
            os.mkdir( os.path.join( outputDir, bgDir ) )
    assert( X.shape == XNew.shape )
    nRow, nCol = XNew.shape
    warnings.filterwarnings(action="ignore", category=UserWarning, message=r".*is a low contrast image" )
    warnings.filterwarnings(action="ignore", category=UserWarning, message=r".*Possible precision loss" )
    for row in range( nRow ):
        skimage.io.imsave( os.path.join( outputDir, bgDir, "frame_bg_{}.jpg".format( row ) ),
                            skimage.util.img_as_ubyte( XNew[ row ].reshape( imgDim ) ) )
        fgBool = np.logical_not( np.isclose( X[ row ], XNew[ row ], atol=threshold ) )
        skimage.io.imsave( os.path.join( outputDir, fgDir, "frame_fg_{}.jpg".format( row ) ),
                            skimage.util.img_as_ubyte( np.where( fgBool, X[ row ], 0.0 ).reshape( imgDim ) ) )

@contextmanager
def createFigureWrapper():
    maxWidth, maxHeight = getScreenRes()
    try:
        yield plt.figure( figsize=( maxWidth/200.0, maxHeight/200.0 ) )
    finally:
        plt.close()

def getScreenRes():
    window = plt.get_current_fig_manager().window
    maxWidth, maxHeight = window.wm_maxsize()
    plt.close()
    return maxWidth, maxHeight

def plotSeparatedImage( frame, background, threshold, figure, dim=imgDim ):
    plt.clf()
    numSubplots = 3
    ax = [ figure.add_subplot( 1, numSubplots, index + 1 ) for index in range( numSubplots ) ]
    ax[ 0 ].set_title( "Original Frame" )
    ax[ 0 ].imshow(frame.reshape( dim ), cmap="gray" )
    ax[ 1 ].set_title( "Background" )
    ax[ 1 ].imshow( background.reshape( dim ), cmap="gray" )
    fgBool = np.logical_not( np.isclose( frame, background, atol=threshold ) )
    ax[ 2 ].set_title( "Foreground" )
    ax[ 2 ].imshow( np.where( fgBool, frame, 0.0 ).reshape( dim ), cmap="gray" )
    plt.show( block=False )
    plt.pause( 0.001 )

if __name__ == "__main__":
    ## convert any image not in .jpg format to .jpg
    imageFilePaths = glob.glob( os.path.join( inputDir, "*" ) )
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
    if os.path.exists( "data.pkl" ):
        os.remove( "data.pkl" )
    with open( "data.pkl", "wb" ) as f:
        pickle.dump( imgData, f )
    # with open( "data.pkl", "rb" ) as f:
    #     data = pickle.load( f )
    # import pdb; pdb.set_trace()