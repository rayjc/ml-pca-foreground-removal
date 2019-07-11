import argparse
import glob
import os

imgDim = [ 120, 160 ]       ## ( 240, 426 ) for 240p on Youtube

class CliConfig:
    def __init__( self ):
        self.config = CliConfig.getCliConfig()
        self.verify()
        self.process()

    @staticmethod
    def getCliConfig():
        mainParser = argparse.ArgumentParser( description="An application using L1 PCA\
                                                             to separate foreground and\
                                                             background by identify moving\
                                                             parts." )
        subParsers = mainParser.add_subparsers( dest="subParser",
                                                help=
                                                "'all': Pre-process frames into\
                                                    .pkl file then run PCA to \
                                                    separate foreground and \
                                                    background. \
                                                'preprocess': Pre-process frames into\
                                                    .pkl file. \
                                                'train': Run PCA to \
                                                    separate foreground and \
                                                    background" )

        allParser = subParsers.add_parser( "all" )
        processParser = subParsers.add_parser( "preprocess" )
        for parser in ( allParser, processParser ):
            parser.add_argument( "-i", "--input", default="input", type=str,
                                    help="Specify directory containing input frames;\
                                            default is 'input'.",
                                    metavar="input_dir" )
            parser.add_argument( "-o", "--output", default="unnamed", type=str,
                                    help="Specify name of output file excluding\
                                             file extension; default is 'unnamed'.",
                                    metavar="output_name" )
            parser.add_argument( "-s", "--size", default=imgDim, type=int, nargs=2,
                                    help="Specify output resolution for all image\
                                             frames; default is 120 by 160.",
                                    metavar=( "img_height", "img_width" ) )
        
        separateParser = subParsers.add_parser( "train" )
        separateParser.add_argument( "-d", "--data", default="*.pkl", type=str,
                                    help="Specify input .pkl file which contains\
                                        a matrix of size nxd (n frames, d pixels);\
                                        default searches current directory.",
                                    metavar="data_filepath" )
        separateParser.add_argument( "-o", "--output", default="", type=str,
                                    help="Specify name of output file excluding\
                                             file extension; default is the same\
                                             name as selected .pkl file.",
                                    metavar="output_name" )
        separateParser.add_argument( "-s", "--size", default=imgDim, type=int, nargs=2,
                                    help="Specify the image resolution of the frames\
                                             in .pkl; default is 120 by 160.",
                                    metavar=( "img_height", "img_width" ) )

        return vars( mainParser.parse_args() )

    def process( self ):
        self.config[ "size" ] = tuple( self.config[ "size" ] )
        self._findPickleFile()

    def verify( self ):
        """ Sanity check for the input CLI arguments """
        assert "." not in self.config[ "output" ],\
                "--output should not exclude any file extension"
        assert len( self.config[ "output" ] ) < 30,\
                "--output should not exceed 30 characters..."
        assert all( val < 3000 for val in self.config[ "size" ] ),\
                "--size seems to be too large..."
        if self.config[ "subParser" ] == "all"\
            or self.config[ "subParser" ] == "preprocess":
            assert os.path.isdir( self.config[ "input" ] )
        if self.config[ "subParser" ] == "train":
            assert "pkl" in self.config[ "data" ] and glob.glob( self.config[ "data" ] ),\
                    "Input data file must be in .pkl format"

    def _findPickleFile( self ):
        """
        Search for .pkl files in current directory if -d option is not used;
        set output file name accordingly.
        """
        if self.config[ "subParser" ] == "train":
            matchedFiles = glob.glob( self.config[ "data" ] )
            if len( matchedFiles ) > 1:
                dataFile = input( "Multiple .pkl files found: {}. Please enter one of the above: "
                                    .format( matchedFiles ) )
                while( dataFile not in matchedFiles ):
                    dataFile = input( "Multiple .pkl files found: {}. Please enter one of the above: "
                            .format( matchedFiles ) )
            else:
                dataFile = matchedFiles[ 0 ]
            self.config[ "data" ] = dataFile
            self.config[ "output" ] = dataFile.split( '.' )[ 0 ]

## Helper functions
def rgb2Grey( rgbImg ):
    """ Given img matrix of sizeMxNx3, return greyscale image matrix of size MxN """
    return np.dot( rgbImg[...,:3], [ 0.2989, 0.5870, 0.1140 ] )