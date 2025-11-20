################################################################
##
## EE40098 Coursework B
##
## File         :  schema.py
## Exercise     :  5
## Author       :  samh25
## Created      :  2025-11-20 (YYYY-MM-DD)
## License      :  MIT
## Description  :  A class representing a schema in a 
##                 genetic algorithm.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import numpy as np

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Schema:

    ############################################################
    ## CONSTRUCTOR
    
    # instantiate a new population
    def __init__(self, gene_index, bit_mask, bit_pattern):

        self.gene_index = gene_index

        # ensure bit mask and pattern are int16
        self.bit_mask = np.uint16(bit_mask)
        self.bit_pattern = np.uint16(bit_pattern)