from .constants import *
from .config import *
from .sequence_clustering import *
from .standard_import import *

########################################################################################################################
### dimensionality reduction and clustering algorithms ###
########################################################################################################################

cluster_algo = hdbscan.HDBSCAN
dim_reduct_algo = umap.UMAP

########################################################################################################################
### dimensionality reduction and clustering algorithms parameters ###
########################################################################################################################

# configure classes for parameter grid generation
# umap
class UmapParaGrid(ParaGrid):

    def _add_parameters(self):

        n_components_min = self._return_dim_pct(2.5)
        n_components_max = self._return_dim_pct(10)
        self.para_grid_generator.add_parameter(CLUSTERING_UMAP_N_COMPONENTS_NAME_STR,
                                               True,
                                               True,
                                               n_components_min,
                                               n_components_max,
                                               10,
                                               None)

        self.para_grid_generator.add_parameter(CLUSTERING_UMAP_N_NEIGHBORS_NAME_STR,
                                               True,
                                               True,
                                               30,
                                               100,
                                               10,
                                               None)

        self.para_grid_generator.add_parameter(CLUSTERING_UMAP_MIN_DIST_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [0])

        self.para_grid_generator.add_parameter(CLUSTERING_UMAP_RANDOM_STATE_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [RNG_SEED])

# hdbscan
class HdbscanParaGrid(ParaGrid):

    def _add_parameters(self):

        min_cluster_size_min = self._return_dim_pct(2.5)
        min_cluster_size_max = self._return_dim_pct(10)
        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR,
                                               True,
                                               True,
                                               min_cluster_size_min,
                                               min_cluster_size_max,
                                               30,
                                               None)

        min_samples_min = self._return_dim_pct(1)
        min_samples_max = self._return_dim_pct(15)
        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR,
                                               True,
                                               True,
                                               min_samples_min,
                                               min_samples_max,
                                               30,
                                               None)

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_GEN_MIN_SPAN_TREE_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [True])

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_CORE_DIST_N_JOBS_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [NUMBER_OF_CORES])

        # self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_CLUSTER_SELECTION_METHOD_NAME_STR,
        #                                        False,
        #                                        False,
        #                                        None,
        #                                        None,
        #                                        None,
        #                                        ['eom', 'leaf'])

        # self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_METRIC_NAME_STR,
        #                                        False,
        #                                        False,
        #                                        None,
        #                                        None,
        #                                        None,
        #                                        ['precomputed'])

########################################################################################################################
### cluster validation function and parameters ###
########################################################################################################################

# clustering validation algos
# hdbscan.validity.validity_index
CLUSTERING_HDBSCAN_VALIDITY_INDEX_NAME_STR = hdbscan.validity.validity_index.__name__
CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_NAME_STR = 'metric'
CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_VALUE_NAME_STR = 'euclidean'
# hdbscan relative_validity_ attribute
CLUSTERING_HDBSCAN_RELATIVE_VALIDITY_ATTRIBUTE_NAME_STR = 'relative_validity_'

# validation attributes tuple
cluster_validation_attributes = (CLUSTERING_HDBSCAN_RELATIVE_VALIDITY_ATTRIBUTE_NAME_STR,)

# cluster validation functions tuple
cluster_validation_functions = (hdbscan.validity.validity_index,)

# cluster validation parameter grids
class ValidityIndex(ParaGrid):

    def _add_parameters(self):

        self.para_grid_generator.add_parameter(CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_NAME_STR,
                                               False,
                                               False,
                                               None,
                                               None,
                                               None,
                                               [CLUSTERING_HDBSCAN_VALIDITY_INDEX_METRIC_VALUE_NAME_STR])

validity_index_grid_generator = ValidityIndex(None)
validity_index_para_grid = validity_index_grid_generator.return_param_grid()

# cluster validation parameter grids tuple
cluster_validation_parameters = (validity_index_para_grid,)

########################################################################################################################
### parallel coordinates field list ###
########################################################################################################################

parallel_coordinates_param_list = [CLUSTERING_UMAP_N_COMPONENTS_NAME_STR, 
                                   CLUSTERING_UMAP_N_NEIGHBORS_NAME_STR, 
                                   CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_NAME_STR, 
                                   CLUSTERING_HDBSCAN_MIN_SAMPLES_NAME_STR]

########################################################################################################################
### cluster 2d plot reducer object ###
########################################################################################################################

reducer_umap = umap.UMAP(n_components=2,
                         random_state=RNG_SEED,
                         verbose = False)

reducer_pca = PCA(n_components=2)