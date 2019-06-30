## 3rd party packages
import imageio
import matplotlib.pyplot as plt
import numpy as np


## TODO: create arg parser module to specify image file path
imgFilePath = "messy_room.jpg"

## TODO: move these to a different file
## Helper functions
def rgb2Grey( rgbImg ):
    return np.dot( rgbImg[...,:3], [ 0.2989, 0.5870, 0.1140 ] )

if __name__ == "__main__":
    ## load .png file into np array
    imgData = imageio.imread( imgFilePath )
    ## only handle monochrome image; TODO: handle rgb
    ## Convert RGB image to monochrome
    if len( imgData.shape ) > 1:
        imgData = rgb2Grey( imgData )
    # print( imgData.shape )
    # plt.imshow(imgData, cmap=plt.cm.gray)
    # plt.show()