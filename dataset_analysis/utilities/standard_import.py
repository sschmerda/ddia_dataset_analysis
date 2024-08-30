# standard
import os
import glob
import time
import random
from collections import defaultdict
from itertools import chain, combinations, combinations_with_replacement, product
import pickle
import warnings
import operator
from abc import ABC
from bs4 import BeautifulSoup
from dataclasses import dataclass
from enum import Enum

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
from numpy.typing import ArrayLike
from numpy.typing import NDArray

# misc
from pympler import asizeof

# var
from IPython.display import display, Markdown
from tqdm import tqdm

# stats libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import iqr
from scipy.stats.contingency import chi2_contingency, association, crosstab
from scipy.stats.contingency import expected_freq
from scipy.spatial.distance import squareform
from scipy.stats import bootstrap
import pingouin as pg

# plotting
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import seaborn as sns
from seaborn.axisgrid import FacetGrid
import plotly.graph_objects as go
from statsmodels.graphics.mosaicplot import mosaic

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
