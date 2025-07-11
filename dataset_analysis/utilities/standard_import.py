# standard
import os
import glob
from pathlib import Path
import time
import random
from collections import defaultdict
from itertools import chain, combinations, combinations_with_replacement, product
import pickle
import operator
from abc import ABC
from bs4 import BeautifulSoup
from dataclasses import dataclass
from enum import Enum
import math
import copy
from functools import wraps, reduce

# typing
from typing import Type
from typing import Any
from typing import Union
from typing import Literal
from typing import List
from typing import Callable
from typing import Iterable
from typing import Dict
from typing import Tuple
from typing import Generator
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import DefaultDict

# misc
from pympler import asizeof
import inflection

# var
from IPython.display import display, Markdown
from tqdm import tqdm

# stats libraries
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
from scipy.stats import iqr
from scipy.stats.contingency import chi2_contingency, association, crosstab
from scipy.stats.contingency import expected_freq
from scipy.spatial.distance import squareform
from scipy.stats import bootstrap
import pingouin as pg

# scaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# R support
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.rinterface_lib.callbacks
r('library(stats)')
r('library(permuco)')

# plotting
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.axes
import seaborn as sns
from seaborn.axisgrid import FacetGrid
import plotly.graph_objects as go
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# warnings 
import warnings
from scipy.stats import DegenerateDataWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# libraries for clustering
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import umap
from sklearn.decomposition import PCA
import hdbscan

# sequence distance libraries
# imported in config file!

# parallelization
from joblib import Parallel, delayed