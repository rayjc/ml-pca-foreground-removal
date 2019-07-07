import glob
import os
import pickle
import shutil

import numpy as np
import skimage.io, skimage.util, skimage.transform

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
    for row in range( nRow ):
        skimage.io.imsave( os.path.join( outputDir, bgDir, "frame_bg_{}.jpg".format( row ) ),
                            skimage.util.img_as_ubyte( XNew[ row ].reshape( imgDim ) ) )
        fgBool = np.logical_not( np.isclose( X[ row ], XNew[ row ], atol=threshold ) )
        skimage.io.imsave( os.path.join( outputDir, fgDir, "frame_fg_{}.jpg".format( row ) ),
                            skimage.util.img_as_ubyte( np.where( fgBool, X[ row ], 0.0 ).reshape( imgDim ) ) )


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