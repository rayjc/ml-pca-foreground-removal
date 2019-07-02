import glob
import imageio
import numpy as np
import os
import pickle
import skimage
import shutil

testImgPath = "messy_room.jpg"
imgDim = ( 240, 426 )       # 240p on Youtube
inputDir = "input"
processedDir = "input-preprocessed"
outputDataFile = "data.pkl"

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
        ## TODO: consider expand normalized greyscale to 8 bit value
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