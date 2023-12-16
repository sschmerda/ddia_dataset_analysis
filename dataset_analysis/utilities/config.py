from .standard_import import *

########################################################################################################################
# sequence distance libraries
# function needs to take 2 strings as the first 2 arguments and return a numeric distance: seq_dist(string_1, string_2)

# from Levenshtein import distance as distance # second fastest implementation in python(mainly written in C)
from polyleven import levenshtein as distance # fastest implementation in python(mainly written in C)

# specify positional and keyword arguments for the distance function(if there are any)
SEQUENCE_DISTANCE_FUNCTION_ARGS = []
SEQUENCE_DISTANCE_FUNCTION_KWARGS = {}
########################################################################################################################

########################################################################################################################
### parallelization options ###

# specify whether the computations should be done in parallel
PARALLELIZE_COMPUTATIONS = True

# specify the number of cores to be utilized
# all cores available will be utilized if value equals -1
NUMBER_OF_CORES = -1
########################################################################################################################

########################################################################################################################
### pandas options ###

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
########################################################################################################################

########################################################################################################################
### seaborn options ###

# set figsize
sns.set(rc = {'figure.figsize':(15,8)})

#sns.set_context('notebook', rc={'font.size':20,'axes.titlesize':30,'axes.labelsize':20, 'xtick.labelsize':16, 'ytick.labelsize':16, 'legend.fontsize':20, 'legend.title_fontsize':20})   
sns.set_style('darkgrid')
marker_config = {'marker':'o',
                 'markerfacecolor':'white', 
                 'markeredgecolor':'black',
                 'markersize':'10'}
marker_config_eval_metric_mean = {'marker':'o',
                                  'markerfacecolor':'red', 
                                  'markeredgecolor':'black',
                                  'markersize':'6'}
########################################################################################################################

########################################################################################################################
### numpy options ###

# turn off an unecessary numpy warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
########################################################################################################################