########################################################################################################################
### sequence distance libraries ###
########################################################################################################################
# function needs to take 2 strings as the first 2 arguments and return a numeric distance: seq_dist(string_1, string_2)

# from Levenshtein import distance as distance # second fastest implementation in python(mainly written in C)
from polyleven import levenshtein as distance # fastest implementation in python(mainly written in C)

# specify positional and keyword arguments for the distance function(if there are any)
SEQUENCE_DISTANCE_FUNCTION_ARGS = []
SEQUENCE_DISTANCE_FUNCTION_KWARGS = {}

########################################################################################################################
### general ###
########################################################################################################################

# specify whether the group field(given the datasets has one) should be ignored when calculating sequence distances.
# sequence distances will be calculated over the entire length of a user's learning activities in the interactions dataframe.
SEQUENCE_DISTANCE_IGNORE_GROUPS = False

########################################################################################################################
### distance matrix plot ###
########################################################################################################################

SEQUENCE_DISTANCE_DISTANCE_MATRIX_PLOT_NORMALIZE_DISTANCE = False
SEQUENCE_DISTANCE_DISTANCE_MATRIX_PLOT_USE_UNIQUE_SEQUENCE_DISTANCES = False
SEQUENCE_DISTANCE_DISTANCE_MATRIX_PLOT_HEIGHT = None
SEQUENCE_DISTANCE_DISTANCE_MATRIX_GROUP_LIST = None
