import argparse


def getCliConfig():
    mainParser = argparse.ArgumentParser( description="python removeForeground.py [all, preprocess]" )
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
                                help="Specify directory containing input frames",
                                metavar="input_dir" )
        parser.add_argument( "-o", "--output", default="", type=str,
                                help="Specify name of output file",
                                metavar="output_name" )
    
    separateParser = subParsers.add_parser( "train" )
    separateParser.add_argument( "-d", "--data", default="*pkl", type=str,
                                help="Specify input .pkl file which contains\
                                    a matrix of size nxd (n frames, d pixels)",
                                metavar="data_filepath" )
    separateParser.add_argument( "-o", "--output", default="", type=str,
                                help="Specify name of output file", metavar="output_name" )

    return mainParser.parse_args()
