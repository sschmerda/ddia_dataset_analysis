from .general_config import *
from ..constants.constants import *
from ..constants.enums import *
from ..standard_import import *
from ..data_classes import ClusterParameter

########################################################################################################################
### param grid base classes ###
########################################################################################################################

def add_param_praefix(param_name: str, 
                      param_type: ParaGridParaType) -> str:

    return f'{param_type.value}{param_name}'

def remove_param_praefix(param_name: str, 
                         param_type: ParaGridParaType) -> str:
        
    return param_name.replace(param_type.value,
                              '',
                              1)
                            
class ParameterGridGenerator():
    """Class for generating a parameter grid used for hyperparameter tuning of the clustering algorithm"""
    def __init__(self,
                 distance_matrix: pd.DataFrame | None,
                 param_type: ParaGridParaType):
        self.distance_matrix = distance_matrix
        self.param_type = param_type

        if self.distance_matrix is not None:
            self.dimensions = distance_matrix.shape[1]
        else:
            self.dimensions = None

        # parameter dict list
        self.parameter_dict_list = []

    def add_parameter(self,
                      param_name: str,
                      is_numeric: bool,
                      is_int: bool,
                      range_min: int | float | None,
                      range_max: int | float | None,
                      n_candidates: int | None,
                      categorical_values: List[str] |  List[int] | List[float] | List[bool] | None) -> None:
        """Adds a parameter to the parameter grid

        Parameters
        ----------
        param_name : str
            The name of the parameter
        is_numeric : bool
            A flag indicating whether the parameter is numeric
        is_int : bool
            A flag indicating whether the parameter is an integer
        range_min : int | float | None
            The minimum value of the parameter
        range_max : int | float | None
            The maximum value of the parameter
        n_candidates : int | None
            The number of possible parameter candidates within the min/max range
        categorical_values : List[str] | List[int] | List[float] | List[bool] | None
            A list of possible categorical values for the parameter
        """
        param_name = add_param_praefix(param_name,
                                       self.param_type)

        param_dict = {CLUSTERING_PARAMETER_TUNING_PARAMETER_NAME_NAME_STR: param_name,
                      CLUSTERING_PARAMETER_TUNING_IS_NUMERIC_NAME_STR: is_numeric,
                      CLUSTERING_PARAMETER_TUNING_IS_INT_NAME_STR: is_int,
                      CLUSTERING_PARAMETER_TUNING_RANGE_MIN_NAME_STR: range_min,
                      CLUSTERING_PARAMETER_TUNING_RANGE_MAX_NAME_STR: range_max,
                      CLUSTERING_PARAMETER_TUNING_NUMBER_CANDIDATES_NAME_STR: n_candidates,
                      CLUSTERING_PARAMETER_TUNING_CATEGORICAL_VALUES_NAME_STR: categorical_values}

        self.parameter_dict_list.append(param_dict)

    def return_parameter_grid(self) -> Iterable[Dict[str, Any]]:
        """Returns a parameter grid for specified parameter values

        Returns
        -------
        Iterable[Dict[str, Any]]
            The parameter grid

        Raises
        ------
        ValueError
            If no input parameter has been specified
        ValueError
            If one or more parameters were misspecified 
        """        
        if not self.parameter_dict_list:
            raise ValueError(CLUSTERING_PARAMETER_TUNING_ERROR_NO_INPUT_PARAMETER_NAME_STR)

        try:
            grid_dict = {}
            for param in self.parameter_dict_list:

                if param[CLUSTERING_PARAMETER_TUNING_IS_NUMERIC_NAME_STR]:

                    if param[CLUSTERING_PARAMETER_TUNING_IS_INT_NAME_STR]:
                        data_type = 'int'
                    else:
                        data_type = 'float'

                    linspace_input = (param[CLUSTERING_PARAMETER_TUNING_RANGE_MIN_NAME_STR], 
                                      param[CLUSTERING_PARAMETER_TUNING_RANGE_MAX_NAME_STR], 
                                      param[CLUSTERING_PARAMETER_TUNING_NUMBER_CANDIDATES_NAME_STR])

                    parameter_range = np.linspace(*linspace_input, 
                                                  dtype=data_type) 
                    parameter_range = np.unique(parameter_range)
                else:
                    parameter_range = param[CLUSTERING_PARAMETER_TUNING_CATEGORICAL_VALUES_NAME_STR]
                
                grid_dict[param[CLUSTERING_PARAMETER_TUNING_PARAMETER_NAME_NAME_STR]] = parameter_range

            param_grid = ParameterGrid(grid_dict)

            return param_grid

        except:
            raise ValueError(CLUSTERING_PARAMETER_TUNING_ERROR_MISSPECIFIED_INPUT_PARAMETER_NAME_STR)
    
    def return_parameters(self) -> tuple[str]:
        """Returns a tuple containing the parameter names of the grid

        Returns
        -------
        tuple
            A tuple containing the parameter names of the parameter grid

        Raises
        ------
        ValueError
            If no input parameter has been specified
        """
        if not self.parameter_dict_list:
            raise ValueError(CLUSTERING_PARAMETER_TUNING_ERROR_NO_INPUT_PARAMETER_NAME_STR)

        return tuple([param_dict[CLUSTERING_PARAMETER_TUNING_PARAMETER_NAME_NAME_STR] for param_dict in self.parameter_dict_list])
    

    def _return_dim_pct(self,
                        pct: int | float) -> int | None:
        """Return a percentage of the dimensions of the distance matrix

        Parameters
        ----------
        pct : int | float
            The percentage 

        Returns
        -------
        int
            Percentage of distance matrix dimensions
        """
        pct_of_dim = None
        if self.distance_matrix is not None:
            pct_of_dim = round(self.dimensions * pct / 100)
        
        return pct_of_dim


class ParaGrid(ABC):
    """Abstract class which allows for building a class for generating a parameter grid used for hyperparameter tuning of the clustering algorithm. 
    Parameter values can be a function of the input dimensions of the respective distance matrix via using the number_samples, number_features attributes."""

    def __init__(self, 
                 distance_matrix: pd.DataFrame | None,
                 param_type: ParaGridParaType):
        """
        Parameters
        ----------
        distance_matrix : pd.DataFrame | None
            The distance matrix used for clustering
        """

        self.distance_matrix = distance_matrix
        self.param_type = param_type
        self.para_grid_generator = ParameterGridGenerator(self.distance_matrix,
                                                          self.param_type)

        if self.distance_matrix is not None:
            self.dimensions = distance_matrix.shape[1]
        else:
            self.dimensions = None

        # add parameter configuration to ParameterGridGenerator object
        self._add_parameters()

    def _add_parameters(self):
        # fill with add_parameter operations for the respective dimensionality reduction or clustering algorithm
        pass

    def return_param_grid(self):
        """Returns a parameter grid for specified parameter values

        Returns
        -------
        Iterable[Dict[str, Any]]
            The parameter grid

        Raises
        ------
        ValueError
            If no input parameter has been specified
        ValueError
            If one or more parameters were misspecified 
        """        

        # return parameter grid
        para_grid = self.para_grid_generator.return_parameter_grid()

        return para_grid
    
    def return_params(self) -> tuple[str]:
        """Returns a tuple containing the names of parameters of the grid

        Returns
        -------
        tuple
            A tuple containing parameter names
        """
        params = self.para_grid_generator.return_parameters()

        return params

########################################################################################################################
### dimensionality reduction and clustering algorithms parameter grids ###
########################################################################################################################

# parameters should be seen in relationship to total number of observations in group (there is a lower bound from preprocessing)!!!

# dimensionality reduction algos
# umap parameter values:

# n_components -> should not be tied to number of observations in order to prevent the curse of dimensionality
DIM_REDUCTION_UMAP_N_COMPONENTS_NAME_STR = 'n_components'
dim_reduction_par_umap_n_components = ClusterParameter(DIM_REDUCTION_UMAP_N_COMPONENTS_NAME_STR,
                                                       True,
                                                       True,
                                                       5,
                                                       20,
                                                       4,
                                                       None)
# n_neighbors
DIM_REDUCTION_UMAP_N_NEIGHBORS_NAME_STR = 'n_neighbors'
dim_reduction_par_umap_n_neighbors = ClusterParameter(DIM_REDUCTION_UMAP_N_NEIGHBORS_NAME_STR,
                                                      True,
                                                      True,
                                                      5,
                                                      15,
                                                      5,
                                                      None)
# min_dist
DIM_REDUCTION_UMAP_MIN_DIST_NAME_STR = 'min_dist'
dim_reduction_par_umap_min_dist = ClusterParameter(DIM_REDUCTION_UMAP_MIN_DIST_NAME_STR,
                                                   False,
                                                   False,
                                                   None,
                                                   None,
                                                   None,
                                                   [0]) # needs to be very small to 0 for clustering
# metric
DIM_REDUCTION_UMAP_METRIC_NAME_STR = 'metric'
dim_reduction_par_umap_metric = ClusterParameter(DIM_REDUCTION_UMAP_METRIC_NAME_STR,
                                                 False,
                                                 False,
                                                 None,
                                                 None,
                                                 None,
                                                 ['precomputed']) # [euclidean | precomputed] -> since we pass dist mat it should be precomputed
# random state
DIM_REDUCTION_UMAP_RANDOM_STATE_NAME_STR = 'random_state'
dim_reduction_par_umap_random_state = ClusterParameter(DIM_REDUCTION_UMAP_RANDOM_STATE_NAME_STR,
                                                       False,
                                                       False,
                                                       None,
                                                       None,
                                                       None,
                                                       [RNG_SEED]) # set to a fixed value for reproducibility
# verbose
DIM_REDUCTION_UMAP_VERBOSE_NAME_STR = 'verbose'
dim_reduction_par_umap_verbose = ClusterParameter(DIM_REDUCTION_UMAP_VERBOSE_NAME_STR,
                                                  False,
                                                  False,
                                                  None,
                                                  None,
                                                  None,
                                                  [False])

# clustering algos
# hdbscan parameter values:

# min_cluster_size
CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR = 'min_cluster_size'
clustering_par_hdbscan_n_components = ClusterParameter(CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR,
                                                       True,
                                                       True,
                                                       5,
                                                       15,
                                                       10,
                                                       None)
# min_samples
CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR = 'min_samples'
CLUSTERING_HDBSCAN_MIN_SAMPLES_MIN_VALUE_AS_PCT_OF_OBSERVATIONS = 5
CLUSTERING_HDBSCAN_MIN_SAMPLES_MAX_VALUE_AS_PCT_OF_OBSERVATIONS = 15
CLUSTERING_HDBSCAN_MIN_SAMPLES_NUMBER_OF_CANDIDATES = 10
# cluster_selection_method
CLUSTERING_HDBSCAN_CLUSTER_SELECTION_METHOD_NAME_STR = 'cluster_selection_method'
CLUSTERING_HDBSCAN_CLUSTER_SELECTION_METHOD_VALUE = ['leaf'] # ['eom', 'leaf']
# metric
CLUSTERING_HDBSCAN_METRIC_NAME_STR = 'metric'
CLUSTERING_HDBSCAN_METRIC_VALUE = ['euclidean'] # ['euclidean', 'precomputed']
# core_dist_n_jobs
CLUSTERING_HDBSCAN_CORE_DIST_N_JOBS_NAME_STR = 'core_dist_n_jobs'
CLUSTERING_HDBSCAN_CORE_DIST_N_JOBS_VALUE = [NUMBER_OF_CORES]
# gen_min_span_tree
CLUSTERING_HDBSCAN_GEN_MIN_SPAN_TREE_NAME_STR = 'gen_min_span_tree'
CLUSTERING_HDBSCAN_GEN_MIN_SPAN_TREE_VALUE = [True] # [True | False]

# configure classes for parameter grid generation
# umap
class DimReductionParaGrid(ParaGrid):

    def _add_parameters(self):

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_N_COMPONENTS_NAME_STR,
                                               CLUSTERING_UMAP_N_COMPONENTS_IS_NUMERIC,
                                               CLUSTERING_UMAP_N_COMPONENTS_IS_INT,
                                               CLUSTERING_UMAP_N_COMPONENTS_MIN_VALUE,
                                               CLUSTERING_UMAP_N_COMPONENTS_MAX_VALUE,
                                               CLUSTERING_UMAP_N_COMPONENTS_NUMBER_OF_CANDIDATES,
                                               CLUSTERING_UMAP_N_COMPONENTS_CATEGORICAL_VALUES)

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_N_NEIGHBORS_NAME_STR,
                                               True,
                                               True,
                                               5,
                                               15,
                                               5,
                                               None)

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_MIN_DIST_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_UMAP_MIN_DIST_NAME_VALUE)

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_METRIC_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_UMAP_METRIC_VALUE)

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_RANDOM_STATE_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_UMAP_RANDOM_STATE_VALUE)

        self.para_grid_generator.add_parameter(DIM_REDUCTION_UMAP_VERBOSE_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_UMAP_VERBOSE_VALUE)

# hdbscan
class ClusteringParaGrid(ParaGrid):

    def _add_parameters(self):

        min_cluster_size_min = self._return_dim_pct(CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_MIN_VALUE_AS_PCT_OF_OBSERVATIONS)
        min_cluster_size_max = self._return_dim_pct(CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_MAX_VALUE_AS_PCT_OF_OBSERVATIONS)
        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR,
                                               True,
                                               True,
                                               min_cluster_size_min,
                                               min_cluster_size_max,
                                               CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NUMBER_OF_CANDIDATES,
                                               None)

        min_samples_min = self._return_dim_pct(CLUSTERING_HDBSCAN_MIN_SAMPLES_MIN_VALUE_AS_PCT_OF_OBSERVATIONS)
        min_samples_max = self._return_dim_pct(CLUSTERING_HDBSCAN_MIN_SAMPLES_MAX_VALUE_AS_PCT_OF_OBSERVATIONS)
        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR,
                                               True,
                                               True,
                                               min_samples_min,
                                               min_samples_max,
                                               CLUSTERING_HDBSCAN_MIN_SAMPLES_NUMBER_OF_CANDIDATES,
                                               None)

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_GEN_MIN_SPAN_TREE_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_HDBSCAN_GEN_MIN_SPAN_TREE_VALUE)

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_CORE_DIST_N_JOBS_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_HDBSCAN_CORE_DIST_N_JOBS_VALUE)

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_CLUSTER_SELECTION_METHOD_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_HDBSCAN_CLUSTER_SELECTION_METHOD_VALUE)

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_METRIC_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               CLUSTERING_HDBSCAN_METRIC_VALUE)


########################################################################################################################
### cluster validation function and parameters ###
########################################################################################################################

# clustering validation algos
# hdbscan.validity.validity_index
CLUSTERING_HDBSCAN_VALIDITY_INDEX_NAME_STR = hdbscan.validity.validity_index.__name__
CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_NAME_STR = 'metric'
CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_VALUE_NAME_STR = 'euclidean' # euclidean or precomputed
# hdbscan relative_validity_ attribute
CLUSTERING_HDBSCAN_RELATIVE_VALIDITY_ATTRIBUTE_NAME_STR = 'relative_validity_'

# cluster validation functions tuple
cluster_validation_functions = (hdbscan.validity.validity_index,)

# validation attributes tuple
cluster_validation_attributes = (CLUSTERING_HDBSCAN_RELATIVE_VALIDITY_ATTRIBUTE_NAME_STR,)

# cluster validation parameter grids -> used in combination with cluster validation functions
# validity index
class ValidityIndex(ParaGrid):

    def _add_parameters(self):

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_VALUE_NAME_STR])

validity_index_grid_generator = ValidityIndex(None,
                                              ParaGridParaType.VALIDATION)
validity_index_para_grid = validity_index_grid_generator.return_param_grid()

# cluster validation parameter grids tuple
cluster_validation_parameters = (validity_index_para_grid,)

########################################################################################################################
### dimensionality reduction and clustering algorithms specification ###
########################################################################################################################

cluster_algo = hdbscan.HDBSCAN
dim_reduct_algo = umap.UMAP

SEQUENCE_DISTANCE_CLUSTERING_NORMALIZE_DISTANCE = False
SEQUENCE_DISTANCE_CLUSTERING_USE_UNIQUE_SEQUENCE_DISTANCES = False

########################################################################################################################
### clustering results selection  ###
########################################################################################################################

CLUSTERING_BEST_RESULT_VALIDATION_METRIC_NAME_STR = CLUSTERING_HDBSCAN_VALIDITY_INDEX_NAME_STR
CLUSTERING_BEST_RESULT_VALIDATION_METRIC_LOWER_IS_BETTER_NAME_STR = False

########################################################################################################################
### hyperparameter tuning matrix plot ###
########################################################################################################################

CLUSTERING_HYPERPARAMETER_TUNING_MATRIX_PLOT_PARAMETER_1_NAME_STR = CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR
CLUSTERING_HYPERPARAMETER_TUNING_MATRIX_PLOT_PARAMETER_2_NAME_STR = CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR

########################################################################################################################
### parallel coordinates plot ###
########################################################################################################################

CLUSTERING_PARALLEL_COORDINATES_PLOT_ADDITIONAL_VALIDATION_METRIC_NAME_STR = CLUSTERING_HDBSCAN_RELATIVE_VALIDITY_ATTRIBUTE_NAME_STR
CLUSTERING_PARALLEL_COORDINATES_PLOT_SELECT_ONLY_BEST_EMBEDDINGS_NAME_STR = False

parallel_coordinates_dim_reduction_param_list = [DIM_REDUCTION_UMAP_N_COMPONENTS_NAME_STR, 
                                                 DIM_REDUCTION_UMAP_N_NEIGHBORS_NAME_STR]
parallel_coordinates_cluster_param_list = [CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR, 
                                           CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR]

########################################################################################################################
### clusters 2d plot ###
########################################################################################################################

CLUSTERING_2D_CLUSTER_PLOT_UMAP_USE_BEST_DIM_REDUCTION_PARAMS_NAME_STR = True
CLUSTERING_2D_CLUSTER_PLOT_PCA_USE_BEST_DIM_REDUCTION_PARAMS_NAME_STR = False

reducer_umap = umap.UMAP
reducer_pca = PCA