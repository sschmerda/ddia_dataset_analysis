from .configs.general_config import *
from .configs.cluster_config import *
from .constants.constants import *
from .standard_import import *
from .validators import *
from .data_classes import *
from .plotting_functions import *

class SequenceDistanceClusters():
    """Performs hyperparameter tuning of a clustering algorithm on a distance matrix"""
    def __init__(self,
                 distance_matrix: pd.DataFrame,
                 dataset_name: str, 
                 group: int,
                 normalize_distance: bool,
                 use_unique_sequence_distances: bool,
                 cluster_function: Callable,
                 dimensionality_reduction_function: Callable | None,
                 cluster_validation_functions: Tuple[Callable] | None,
                 cluster_param_grid: Iterable[Dict[str, Any]],
                 dimensionality_reduction_param_grid: Iterable[Dict[str, Any]] | None,
                 cluster_validation_parameters: Tuple[ParameterGrid] | None,
                 cluster_validation_attributes: Tuple[str] | None,
                 parallelize_computation: bool):

        self._cluster_entity_ids = np.array(distance_matrix.columns)
        self._distance_matrix = distance_matrix.values.astype('double')
        self._dataset_name = dataset_name
        self._group = group
        self._normalize_distance = normalize_distance
        self._use_unique_sequence_distances = use_unique_sequence_distances
        self._cluster_function = cluster_function
        self._dimensionality_reduction_function = dimensionality_reduction_function
        self._cluster_validation_functions = cluster_validation_functions
        self._cluster_param_grid = cluster_param_grid
        self._dimensionality_reduction_param_grid = dimensionality_reduction_param_grid
        self._cluster_validation_parameters = cluster_validation_parameters
        self._cluster_validation_attributes = cluster_validation_attributes
        self._parallelize_computation = parallelize_computation


    def _return_dimensionality_reduction_results_dict(self,
                                                      has_dim_reduction: bool,
                                                      reducer_name: str | None,
                                                      dimensionality_reduction_params: dict | None) -> dict:

        dimensionality_reduction_results = {CLUSTERING_PARAMETER_TUNING_DIMENSIONALITY_REDUCED_NAME_STR: has_dim_reduction,
                                            CLUSTERING_PARAMETER_TUNING_DIM_REDUCTION_ALGORITHM_NAME_NAME_STR: reducer_name}

        if dimensionality_reduction_params:
            dimensionality_reduction_results = dimensionality_reduction_results | dimensionality_reduction_params

        return dimensionality_reduction_results 

    def _return_dimensionality_reduction_results(self,
                                                 distance_matrix: np.ndarray,
                                                 dimensionality_reduction_params: dict) -> Tuple[pd.DataFrame, dict]:

        dimensionality_reduction_params_praefix_removed = ({remove_param_praefix(k, ParaGridParaType.DIM_REDUCTION): v 
                                                            for k, v in dimensionality_reduction_params.items()})
        reducer = self._dimensionality_reduction_function(**dimensionality_reduction_params_praefix_removed)
        reduced_matrix = reducer.fit_transform(distance_matrix).astype('double')
        has_dim_reduction = True
        reducer_name = self._dimensionality_reduction_function.__name__

        dimensionality_reduction_results = self._return_dimensionality_reduction_results_dict(has_dim_reduction,
                                                                                              reducer_name,
                                                                                              dimensionality_reduction_params)

        return reduced_matrix, dimensionality_reduction_results

    def _return_cluster_results_dict(self,
                                     algorithm_name: str,
                                     number_clusters: int,
                                     percentage_clustered: float,
                                     smallest_cluster_size: int,
                                     cluster_labels: NDArray[np.int_],
                                     cluster_entity_ids: NDArray[np.int_],
                                     clustered: NDArray[np.int_],
                                     cluster_params: dict) -> dict:

        algorithm_name = {CLUSTERING_PARAMETER_TUNING_ALGORITHM_NAME_NAME_STR: algorithm_name}

        cluster_results_dict = {CLUSTERING_NUMBER_CLUSTERS_NAME_STR: number_clusters,
                                CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR: percentage_clustered,
                                CLUSTERING_SMALLEST_CLUSTER_SIZE_NAME_STR: smallest_cluster_size,
                                CLUSTERING_CLUSTER_LABELS_NAME_STR: cluster_labels,
                                CLUSTERING_CLUSTER_ENTITY_IDS_NAME_STR: cluster_entity_ids,
                                CLUSTERING_CLUSTER_CLUSTERED_NAME_STR: clustered}

        cluster_results_dict = algorithm_name | cluster_params | cluster_results_dict

        return cluster_results_dict
    
    def _return_cluster_validation_dict(self,
                                        distance_matrix: np.ndarray,
                                        cluster_labels: np.ndarray,
                                        clusterer: BaseEstimator) -> dict:

        cluster_validation_dict = {}

        if self._cluster_validation_attributes:
            for attribute in self._cluster_validation_attributes:
                score = getattr(clusterer, attribute)
                score_name = attribute

                cluster_validation_dict[score_name] = score

        if self._cluster_validation_functions:
            if self._cluster_validation_parameters:
                cluster_val_iter = zip(self._cluster_validation_functions, self._cluster_validation_parameters)
            else:
                cluster_val_iter = zip(self._cluster_validation_functions, [ParameterGrid({})]*len(self._cluster_validation_functions))

            for func, para_grid in cluster_val_iter:
                para_dict = list(para_grid)[0]
                para_dict_praefix_removed = ({remove_param_praefix(k, ParaGridParaType.VALIDATION): v 
                                             for k, v in para_dict.items()})
                score = func(distance_matrix, 
                             cluster_labels,
                             **para_dict_praefix_removed)
                score_name = func.__name__

                cluster_validation_dict[score_name] = score

        return cluster_validation_dict
    
    def _return_cluster_base_dict(self,
                                  normalize_distance: bool,
                                  use_unique_sequence_distances: bool) -> dict:
        if use_unique_sequence_distances:
            cluster_entity = CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_SEQUENCE_VALUE_NAME_STR
        else:
            cluster_entity = CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_USER_VALUE_NAME_STR

        cluster_base_dict = {CLUSTERING_CLUSTER_SEQ_DIST_IS_NORMALIZED_NAME_STR: normalize_distance,
                             CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_NAME_STR: cluster_entity}
        
        return cluster_base_dict

    def _return_cluster_results(self,
                                matrix: np.ndarray,
                                cluster_params: dict,
                                dimensionality_reduction_results: dict) -> dict:
        cluster_params_praefix_removed = ({remove_param_praefix(k, ParaGridParaType.CLUSTERING): v 
                                          for k, v in cluster_params.items()})
        clusterer = self._cluster_function(**cluster_params_praefix_removed)
        clusterer.fit(matrix)

        cluster_labels = clusterer.labels_
        clustered = (cluster_labels >= 0)
        n_clusters = len(np.unique(cluster_labels[clustered]))
        percentage_clustered = sum(clustered) / len(cluster_labels) * 100
        cluster_size_array = np.unique(cluster_labels[clustered], return_counts=True)[1]
        if cluster_size_array.size:
            smallest_cluster_size = min(cluster_size_array)
        else:
            smallest_cluster_size = 0
        algo_name = clusterer.__class__.__name__
        
        cluster_results_dict = self._return_cluster_results_dict(algo_name,
                                                                 n_clusters,
                                                                 percentage_clustered,
                                                                 smallest_cluster_size,
                                                                 cluster_labels,
                                                                 self._cluster_entity_ids,
                                                                 clustered,
                                                                 cluster_params)

        cluster_validation_dict = self._return_cluster_validation_dict(matrix,
                                                                       cluster_labels,
                                                                       clusterer)
        
        cluster_base_dict = self._return_cluster_base_dict(self._normalize_distance,
                                                           self._use_unique_sequence_distances)
        
        result_dict = cluster_base_dict | dimensionality_reduction_results | cluster_results_dict | cluster_validation_dict

        return result_dict

    def _do_clustering_parameter_grid_search(self,
                                             matrix: np.ndarray,
                                             dimensionality_reduction_results: dict) -> List[Dict]:

        if self._parallelize_computation:
            results = (Parallel(n_jobs=NUMBER_OF_CORES)
                       (delayed(self._return_cluster_results)(matrix, params, dimensionality_reduction_results) 
                                for params in self._cluster_param_grid))
        else:
            results = []
            for params in self._cluster_param_grid:
                results.append(self._return_cluster_results(matrix,
                                                            params,
                                                            dimensionality_reduction_results))
        
        return results

    def return_cluster_parameter_tuning_results(self) -> pd.DataFrame:
        """Performs hyperparameter tuning for the specified clustering algorithm taking a distance matrix as input and
        returns the results in a pandas dataframe

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the hyperparameter tuning cluster results
        """

        if self._dimensionality_reduction_function:
            matrices = (self._return_dimensionality_reduction_results(self._distance_matrix,
                                                                      dim_reduct_params) 
                        for dim_reduct_params in self._dimensionality_reduction_param_grid)
        else:
            dimensionality_reduction_results = self._return_dimensionality_reduction_results_dict(False,
                                                                                                  None,
                                                                                                  None)
            matrices = ((self._distance_matrix, dimensionality_reduction_results),)

        results_list = []
        for matrix, dimensionality_reduction_results in matrices:
            results = self._do_clustering_parameter_grid_search(matrix,
                                                                dimensionality_reduction_results)
            results_list.extend(results)

        parameter_df = pd.DataFrame(results_list)

        parameter_df.insert(0,
                            DATASET_NAME_FIELD_NAME_STR,
                            self._dataset_name)
        parameter_df.insert(1,
                            GROUP_FIELD_NAME_STR,
                            self._group)

        return parameter_df

class SequenceDistanceClustersPerGroup():
    """Performs hyperparameter tuning of a clustering algorithm on distance matrices of all groups"""

    def __init__(self,
                 dataset_name: str,
                 normalize_distance: bool,
                 use_unique_sequence_distances: bool,
                 sequence_distance_analytics: Any,
                 cluster_function: Callable,
                 dimensionality_reduction_function: Callable | None,
                 cluster_validation_functions: Tuple[Callable] | None,
                 cluster_param_grid_generator: Type[ParaGrid],
                 dimensionality_reduction_param_grid_generator: Type[ParaGrid] | None,
                 cluster_validation_parameters: Tuple[ParameterGrid] | None,
                 cluster_validation_attributes: Tuple[str] | None,
                 parallelize_computation: bool):

        self.dataset_name = dataset_name
        self.normalize_distance = normalize_distance
        self.use_unique_sequence_distances = use_unique_sequence_distances
        self.sequence_distance_analytics = copy.deepcopy(sequence_distance_analytics)
        self.cluster_function = cluster_function
        self.dimensionality_reduction_function = dimensionality_reduction_function
        self.cluster_validation_functions = cluster_validation_functions
        self.cluster_param_grid_generator = cluster_param_grid_generator
        self.dimensionality_reduction_param_grid_generator = dimensionality_reduction_param_grid_generator
        self.cluster_validation_parameters = cluster_validation_parameters
        self.cluster_validation_attributes = cluster_validation_attributes
        self.parallelize_computation = parallelize_computation

        # data to be calculated
        self.cluster_results_per_group = None

        # groups
        self.groups = self.sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group[GROUP_FIELD_NAME_STR].unique()

        # cluster results field list
        self._cluster_results_list = [CLUSTERING_CLUSTER_LABELS_NAME_STR, 
                                      CLUSTERING_CLUSTER_ENTITY_IDS_NAME_STR,
                                      CLUSTERING_CLUSTER_CLUSTERED_NAME_STR,
                                      CLUSTERING_NUMBER_CLUSTERS_NAME_STR,
                                      CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR,
                                      CLUSTERING_SMALLEST_CLUSTER_SIZE_NAME_STR,
                                      CLUSTERING_CLUSTER_SEQ_DIST_IS_NORMALIZED_NAME_STR,
                                      CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_NAME_STR]


        # cluster entities
        if self.use_unique_sequence_distances:
            self.seq_dist_entity = CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_SEQUENCE_VALUE_NAME_STR
        else:
            self.seq_dist_entity = CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_USER_VALUE_NAME_STR

        # sequence counts per group dict
        seq_count_field_list = [LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
                                LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR]
        self.seq_count_dict = self.sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR)[seq_count_field_list].first().to_dict()
        
        # user - sequence_id mapping per group dict
        self.user_seq_mapping_dict_per_group = self._return_user_id_seq_id_mapping_per_group_dict()

        # algorithm names
        self.cluster_algo_name = return_value_if_not_none(self.cluster_function,
                                                          self._return_object_name(self.cluster_function),
                                                          None,
                                                          True,
                                                          CLUSTERING_PARAMETER_TUNING_ERROR_NO_CLUSTER_CLUSTER_NAME_STR)

        self.dimensionality_reduction_algo_name = return_value_if_not_none(self.dimensionality_reduction_function,
                                                                           self._return_object_name(self.dimensionality_reduction_function),
                                                                           None,
                                                                           False,
                                                                           None)

        self.cluster_validation_algo_names = []
        if self.cluster_validation_functions:
            self.cluster_validation_algo_names.extend([self._return_object_name(i) for i in self.cluster_validation_functions])
        else:
            self.cluster_validation_algo_names.extend([])

        if self.cluster_validation_attributes:
            self.cluster_validation_algo_names.extend(list(self.cluster_validation_attributes))
        else:
            self.cluster_validation_algo_names.extend([])

        # algorithm parameters
        if self.cluster_param_grid_generator:
            self.cluster_para_fields = list(self.cluster_param_grid_generator(None, 
                                                                              ParaGridParaType.CLUSTERING)
                                                .return_params())
        else:
            self.cluster_para_fields = None

        if self.dimensionality_reduction_param_grid_generator:
            self.dim_reduct_para_fields = list(self.dimensionality_reduction_param_grid_generator(None,
                                                                                                  ParaGridParaType.DIM_REDUCTION)
                                                   .return_params())
        else:
            self.dim_reduct_para_fields = None
    
    def cluster_sequences_per_group(self) -> None:    

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            cluster_results_list = [self._cluster_sequences(group, self.normalize_distance, self.use_unique_sequence_distances) for group in tqdm(self.groups)]
            
            self.cluster_results_per_group = pd.concat(cluster_results_list)

    def return_best_cluster_results_per_group_df(self,
                                                 cluster_validation_metric: str,
                                                 cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        cluster_results_per_group_df = self._return_best_cluster_results_per_group_df(cluster_validation_metric,
                                                                                      cluster_validation_lower_is_better)
        return cluster_results_per_group_df

    def return_cluster_results_per_group_df(self) -> pd.DataFrame:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        return self.cluster_results_per_group

    def return_sequence_cluster_per_group_df(self,
                                             cluster_validation_metric: str,
                                             cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        sequence_cluster_per_group_df = self._return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                   cluster_validation_lower_is_better)

        return sequence_cluster_per_group_df

    def return_learning_activity_sequence_stats_per_group(self,
                                                          cluster_validation_metric: str,
                                                          cluster_validation_lower_is_better: bool,
                                                          return_unique_seq_stats: bool) -> pd.DataFrame:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        sequence_cluster_per_group_df = self._return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                   cluster_validation_lower_is_better)

        if return_unique_seq_stats:
            learning_activity_sequence_stats_per_group = self._add_cluster_id_to_sequence_stats_df(self.sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group,
                                                                                                   sequence_cluster_per_group_df)
        else:
            learning_activity_sequence_stats_per_group = self._add_cluster_id_to_sequence_stats_df(self.sequence_distance_analytics.learning_activity_sequence_stats_per_group,
                                                                                                   sequence_cluster_per_group_df)

        return learning_activity_sequence_stats_per_group

    def print_number_sequences_per_cluster_per_group(self,
                                                     cluster_validation_metric: str,
                                                     cluster_validation_lower_is_better: bool) -> None:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)
                                                     
        sequence_cluster_per_group_df = self._return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                   cluster_validation_lower_is_better)

        n_seq_per_cluster_df = self._return_n_sequences_per_cluster_per_group_df(sequence_cluster_per_group_df)

        for seq_count_type in [CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
                               CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR]:
            seq_per_cluster_count = n_seq_per_cluster_df.pivot(columns=CLUSTER_FIELD_NAME_STR, 
                                                               index=GROUP_FIELD_NAME_STR, 
                                                               values=seq_count_type).fillna(0).astype(int)
            seq_per_cluster_count[CLUSTERING_TOTAL_NAME_STR] = seq_per_cluster_count.apply(lambda x: sum(x), axis=1)

            print(STAR_STRING)
            print(seq_count_type)
            print(STAR_STRING)
            print(DASH_STRING)
            print(seq_per_cluster_count)
            print(DASH_STRING)
            print(' ')

    def add_cluster_result_to_results_tables(self,
                                             result_tables: Type[Any],
                                             cluster_validation_metric: str,
                                             cluster_validation_lower_is_better: bool) -> None:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)
        
        cluster_results_per_group_df = self._return_best_cluster_results_per_group_df(cluster_validation_metric,
                                                                                      cluster_validation_lower_is_better)

        sequence_cluster_per_group_df = self._return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                   cluster_validation_lower_is_better)

        unique_learning_activity_sequence_stats_per_group = self._add_cluster_id_to_sequence_stats_df(self.sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group,
                                                                                                      sequence_cluster_per_group_df)
        learning_activity_sequence_stats_per_group = self._add_cluster_id_to_sequence_stats_df(self.sequence_distance_analytics.learning_activity_sequence_stats_per_group,
                                                                                               sequence_cluster_per_group_df)
        # add data to results_table
        result_tables.best_cluster_results_per_group_df = cluster_results_per_group_df.copy()
        result_tables.sequence_cluster_per_group_df = sequence_cluster_per_group_df.copy()
        result_tables.unique_learning_activity_sequence_stats_per_group = unique_learning_activity_sequence_stats_per_group.copy()
        result_tables.learning_activity_sequence_stats_per_group = learning_activity_sequence_stats_per_group.copy()

    def plot_cluster_hyperparameter_tuning_matrix(self,
                                                  cluster_param_1: str,
                                                  cluster_param_2: str,
                                                  cluster_validation_metric: str,
                                                  cluster_validation_lower_is_better: bool) -> None:

        cluster_param_1 = add_param_praefix(cluster_param_1, ParaGridParaType.CLUSTERING)
        cluster_param_2 = add_param_praefix(cluster_param_2, ParaGridParaType.CLUSTERING)

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_param_1,
                                    self.cluster_para_fields)
        _ = check_value_in_iterable(cluster_param_2,
                                    self.cluster_para_fields)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)
        

        data_long_all_group, algo_params_dict, cluster_validation_results_dict, annotation_dict = (
            self._calc_hyperparameter_tuning_matrix_plotting_data(cluster_param_1,
                                                                  cluster_param_2,
                                                                  cluster_validation_metric,
                                                                  cluster_validation_lower_is_better))

        seq_count_per_cluster_per_group_dict = self._return_n_sequences_per_cluster_per_group_dict(cluster_validation_metric,
                                                                                                   cluster_validation_lower_is_better)

        # Create a custom colormap
        colors = [SEABORN_HEATMAP_ANNOTATION_COLOR, 'white']
        cmap = LinearSegmentedColormap.from_list('custom', colors)

        g = sns.FacetGrid(data_long_all_group, 
                          col=CLUSTERING_PARAMETER_TUNING_RESULT_METRIC_NAME_STR,
                          row=GROUP_FIELD_NAME_STR,
                          sharex=False,
                          sharey=False,
                          height=SEABORN_FIGURE_LEVEL_HEIGHT_CLUSTER_PARAMETER_TUNING,
                          aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE)
        g.map_dataframe(draw_heatmap,
                        False,
                        True,
                        cluster_param_1, 
                        cluster_param_2, 
                        CLUSTERING_PARAMETER_TUNING_RESULT_VALUE_NAME_STR,
                        annot=True,
                        fmt='0.2f',
                        linewidth=SEABORN_HEATMAP_LINEWIDTH,
                        annot_kws={'fontsize': SEABORN_HEATMAP_ANNOTATION_FONTSIZE},
                        cmap=SEABORN_HEATMAP_CMAP,
                        xticklabels=True, 
                        yticklabels=True)

        # get figure background color
        facecolor=plt.gcf().get_facecolor()
        for ax in g.axes.flat:
            # set aspect of all axis
            ax.set_aspect('equal','box')
            # set background color of axis instance
            ax.set_facecolor(facecolor)

        # loop over rows and columns of facet grid and apply annotations
        row_names = g.row_names
        for row_index, ax_row in enumerate(g.axes):
            row_name = row_names[row_index]
            annotation_vars = annotation_dict[row_name]
            for column_index, ax in enumerate(ax_row):

                cluster_param_1 = remove_param_praefix(cluster_param_1, ParaGridParaType.CLUSTERING)
                cluster_param_2 = remove_param_praefix(cluster_param_2, ParaGridParaType.CLUSTERING)
                
                n_sequences = self.seq_count_dict[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR][row_name]
                n_unique_sequences = self.seq_count_dict[LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR][row_name]

                algo_params = algo_params_dict[row_name]
                cluster_results = cluster_validation_results_dict[row_name]
                seq_count_per_cluster = seq_count_per_cluster_per_group_dict[row_name]

                param_str = self._return_hyperparameter_tuning_string(row_name,
                                                                      self.normalize_distance,
                                                                      self.seq_dist_entity,
                                                                      n_sequences,
                                                                      n_unique_sequences,
                                                                      algo_params.cluster_algo_name,
                                                                      algo_params.dim_reduction_algo_name,
                                                                      cluster_param_1,
                                                                      cluster_param_2,
                                                                      algo_params.cluster_param_series,
                                                                      algo_params.dim_reduction_param_series,
                                                                      cluster_results.cluster_validation_metric_name,
                                                                      cluster_results.cluster_validation_metric_optimum_value,
                                                                      cluster_results.number_clusters,
                                                                      cluster_results.percentage_clustered,
                                                                      seq_count_per_cluster)
                # add str on left side of first column in facet grid
                if column_index == 0:
                    ax.annotate(param_str, 
                                xy=(-0.5, 0.5), 
                                xycoords='axes fraction', 
                                fontsize=16,
                                va='center', 
                                annotation_clip=False)

                plt.tight_layout()

                sns.heatmap(annotation_vars.masked_data_df, 
                            cmap=cmap, 
                            cbar=False, 
                            mask=annotation_vars.masked_data.mask,
                            xticklabels=True, 
                            yticklabels=True,
                            ax=ax)

                ax.set_xlabel(cluster_param_1)
                ax.set_ylabel(cluster_param_2)
                            
                # Highlight the maximum value tiles
                for field, value in annotation_vars.optimum_values_positions:
                    ax.add_patch(patches.Rectangle((value, field), 1, 1, fill=False, edgecolor='black', linewidth=2))

                # Set color of maximum value annotations to black
                heatmap_value_iterator = ax.texts
                for n, text in enumerate(heatmap_value_iterator):
                    if n in annotation_vars.optimum_values_idx:
                        text.set_color('black')

    def plot_cluster_hyperparameter_tuning_parallel_coordinates(self,
                                                                dim_reduction_param_list: List[str],
                                                                cluster_param_list: List[str],
                                                                groups_to_include: list[int] | None,
                                                                cluster_validation_metric: str,
                                                                additional_cluster_validation_metrics: str | list[str] | None,
                                                                cluster_validation_lower_is_better: bool,
                                                                select_only_best_embedding: bool) -> None:

        dim_reduction_param_list = [add_param_praefix(par, ParaGridParaType.DIM_REDUCTION) for par in dim_reduction_param_list]
        cluster_param_list = [add_param_praefix(par, ParaGridParaType.CLUSTERING) for par in cluster_param_list]
        param_list = dim_reduction_param_list + cluster_param_list

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = [check_value_in_iterable(param, self.cluster_results_per_group.columns) for param in param_list]

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        if isinstance(additional_cluster_validation_metrics, str):
            additional_cluster_validation_metrics = [additional_cluster_validation_metrics]
        if additional_cluster_validation_metrics is None:
            additional_cluster_validation_metrics = []
        _ = [check_value_in_iterable(val_metric, self.cluster_validation_algo_names) for val_metric in additional_cluster_validation_metrics]

        result_fields = [CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR, CLUSTERING_NUMBER_CLUSTERS_NAME_STR]
        validation_fields = [cluster_validation_metric] + additional_cluster_validation_metrics
        fields = param_list + validation_fields + result_fields

        print(STAR_STRING)
        print(CLUSTERING_PARAMETER_TUNING_PARALLEL_COORDINATES_PLOT_TITLE_NAME_STR)
        print(STAR_STRING)
        print(' ')
        print(DASH_STRING)
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_IS_NORMALIZED_TITLE_NAME_STR}: {self.normalize_distance}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_ENTITIES_TITLE_NAME_STR}: {self.seq_dist_entity}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_TITLE_NAME_STR}{cluster_validation_metric}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_LOWER_IS_BETTER_TITLE_NAME_STR}{cluster_validation_lower_is_better}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SELECT_ONLY_BEST_EMBEDDING_TITLE_NAME_STR}{select_only_best_embedding}')
        print(DASH_STRING)
        print('\n')
        for group, data in self.cluster_results_per_group.groupby(GROUP_FIELD_NAME_STR):

            if groups_to_include is not None:
                if group not in groups_to_include:
                    continue

            if select_only_best_embedding:
                data = self._select_best_cluster_results_embedding(data,
                                                                   cluster_validation_metric,
                                                                   cluster_validation_lower_is_better)
            data = data[fields]

            dimensions_data_list = []
            for field in fields:
                values = data[field]
                label = field
                min_val = values.min()
                max_val = values.max()
                val_range = [min_val, max_val]

                coordinate_data = dict(range = val_range,
                                       label = label, 
                                       values = values)
                dimensions_data_list.append(coordinate_data)
            
            line_dict = dict(color = data[cluster_validation_metric],
                             colorscale = PLOTLY_PARALLEL_COORDINATES_COLORSCALE,
                             showscale = True,
                             colorbar = dict(title = cluster_validation_metric))

            print(DASH_STRING)
            print(f'{GROUP_FIELD_NAME_STR}: {group}')
            fig = go.Figure(data=go.Parcoords(line = line_dict,
                                              dimensions = list(dimensions_data_list)))
            
            fig.update_layout(width=PLOTLY_PARALLEL_COORDINATES_FIGURE_WIDTH,
                              height=PLOTLY_PARALLEL_COORDINATES_FIGURE_HEIGHT)
            fig.show()
            print('\n')
    
    def plot_clusters_per_group_2d(self,
                                   cluster_validation_metric: str,
                                   cluster_validation_lower_is_better: bool,
                                   reducer: Callable,
                                   use_best_dimension_reduction_params: bool) -> None:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        cluster_results_dict = self._return_cluster_results_per_group_dict(cluster_validation_metric,
                                                                           cluster_validation_lower_is_better)

        embeddings_2d_dfs = [self._calc_cluster_per_group_plotting_data(group, 
                                                                        cluster_results_dict[group], 
                                                                        reducer, 
                                                                        use_best_dimension_reduction_params) for group in self.groups]

        embeddings_2d_per_group_dfs = pd.concat(embeddings_2d_dfs)
        max_n_clusters = np.max([result.number_clusters for result in cluster_results_dict.values()])

        n_cols = set_facet_grid_column_number(embeddings_2d_per_group_dfs[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)
        print(STAR_STRING)
        print(f'{CLUSTERING_RESULT_BEST_CLUSTER_RESULT_TITLE_NAME_STR}')
        print(STAR_STRING)
        print(' ')
        print(DASH_STRING)
        print(f'    - {CLUSTERING_RESULT_DIMENSIONALITY_REDUCER_TITLE_NAME_STR}{reducer.__name__}')
        print(f'    - {CLUSTERING_RESULT_UNCLUSTERED_COLOR_TITLE_NAME_STR}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_IS_NORMALIZED_TITLE_NAME_STR}{self.normalize_distance}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_ENTITIES_TITLE_NAME_STR}{self.seq_dist_entity}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_TITLE_NAME_STR}{cluster_validation_metric}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_LOWER_IS_BETTER_TITLE_NAME_STR}{cluster_validation_lower_is_better}')
        print(DASH_STRING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = sns.FacetGrid(embeddings_2d_per_group_dfs, 
                              col=GROUP_FIELD_NAME_STR, 
                              col_wrap=n_cols, 
                              sharex=False,
                              sharey=False,
                              height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                              aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE)
            g.map_dataframe(sns.scatterplot, 
                            x=CLUSTERING_2D_PLOT_NOT_CLUSTERED_X_AXIS_FIELD_NAME_STR, 
                            y=CLUSTERING_2D_PLOT_NOT_CLUSTERED_Y_AXIS_FIELD_NAME_STR,
                            color='black',
                            alpha=SEABORN_POINT_ALPHA_FACET_CLUSTER_2D,
                            s=SEABORN_POINT_SIZE_FACET_CLUSTER_2D)
            g.map_dataframe(sns.scatterplot, 
                            x=CLUSTERING_2D_PLOT_CLUSTERED_X_AXIS_FIELD_NAME_STR, 
                            y=CLUSTERING_2D_PLOT_CLUSTERED_Y_AXIS_FIELD_NAME_STR,
                            hue=CLUSTER_FIELD_NAME_STR,
                            palette=return_color_palette(max_n_clusters),
                            alpha=SEABORN_POINT_ALPHA_FACET_CLUSTER_2D,
                            s=SEABORN_POINT_SIZE_FACET_CLUSTER_2D,
                            edgecolor=SEABORN_POINT_EDGECOLOR)
            g.add_legend(title=CLUSTER_FIELD_NAME_STR,
                         frameon=True,
                         markerscale=2)
            sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 0.75))
            g.set(xlabel=CLUSTERING_2D_PLOT_X_AXIS_LABEL_NAME_STR, 
                  ylabel=CLUSTERING_2D_PLOT_Y_AXIS_LABEL_NAME_STR)
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            g.figure.subplots_adjust(top=0.95)
            g.figure.tight_layout(rect=[0, 0.03, 1, 0.98]);
            plt.show(g)
    
    def plot_number_of_sequences_per_cluster_per_group(self,
                                                       cluster_validation_metric: str,
                                                       cluster_validation_lower_is_better: bool) -> None:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        sequence_cluster_per_group_df = self._return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                   cluster_validation_lower_is_better)

        n_seq_per_cluster_df = self._return_n_sequences_per_cluster_per_group_df(sequence_cluster_per_group_df)

        n_seq_per_cluster_df_long = pd.melt(n_seq_per_cluster_df,
                                            [GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR],
                                            [CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR,
                                             CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR],
                                            CLUSTERING_NUMBER_OF_SEQUENCES_PER_CLUSTER_VAR_FIELD_NAME_STR,
                                            CLUSTERING_NUMBER_OF_SEQUENCES_PER_CLUSTER_VAL_FIELD_NAME_STR)

        n_cols = set_facet_grid_column_number(n_seq_per_cluster_df_long[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        print(STAR_STRING)
        print(f'{CLUSTERING_RESULT_N_SEQ_PER_CLUSTER_TITLE_NAME_STR}')
        print(STAR_STRING)
        print(' ')
        print(DASH_STRING)
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_IS_NORMALIZED_TITLE_NAME_STR}{self.normalize_distance}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_ENTITIES_TITLE_NAME_STR}{self.seq_dist_entity}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_TITLE_NAME_STR}{cluster_validation_metric}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_LOWER_IS_BETTER_TITLE_NAME_STR}{cluster_validation_lower_is_better}')
        print(DASH_STRING)
        g = sns.catplot(data=n_seq_per_cluster_df_long,
                        x=CLUSTERING_NUMBER_OF_SEQUENCES_PER_CLUSTER_VAL_FIELD_NAME_STR,
                        y=CLUSTER_FIELD_NAME_STR,
                        hue=CLUSTERING_NUMBER_OF_SEQUENCES_PER_CLUSTER_VAR_FIELD_NAME_STR,
                        col=GROUP_FIELD_NAME_STR,
                        orient='h',
                        col_wrap=n_cols, 
                        kind='bar',
                        height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                        aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE,
                        sharex=True,
                        sharey=True)
        plt.tight_layout()
        g.add_legend(title=CLUSTERING_NUMBER_OF_SEQUENCES_PER_CLUSTER_VAR_FIELD_NAME_STR,
                    frameon=True)
        plt.show(g)

    def plot_cluster_stats_per_group(self,
                                     cluster_validation_metric: str,
                                     cluster_validation_lower_is_better: bool) -> None:
        
        cluster_stats_per_group_df = self._calc_cluster_stats_per_group_plotting_data(cluster_validation_metric,
                                                                                      cluster_validation_lower_is_better)

        axis_lim_pct = return_axis_limits(cluster_stats_per_group_df[CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR],
                                          True,
                                          pct_is_ratio=False)
        axis_lim_pct_dict = dict(xlim=axis_lim_pct)

        field_list = [CLUSTERING_NUMBER_CLUSTERS_NAME_STR, CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR, CLUSTERING_SMALLEST_CLUSTER_SIZE_NAME_STR]
        title_list = [CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR, CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR, CLUSTERING_SMALLEST_CLUSTER_SIZE_FIELD_NAME_STR]
        axis_lim_dict_list = [dict(), axis_lim_pct_dict, dict()]
        field_title_iterator = zip(field_list, title_list, axis_lim_dict_list)
        n_groups = len(np.unique(self.groups))

        print(STAR_STRING)
        print(f'{CLUSTERING_RESULT_CLUSTER_STATS_TITLE_NAME_STR}')
        print(STAR_STRING)
        print(' ')
        print(DASH_STRING)
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_IS_NORMALIZED_TITLE_NAME_STR}{self.normalize_distance}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_SEQ_DIST_ENTITIES_TITLE_NAME_STR}{self.seq_dist_entity}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_TITLE_NAME_STR}{cluster_validation_metric}')
        print(f'    - {CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_LOWER_IS_BETTER_TITLE_NAME_STR}{cluster_validation_lower_is_better}')
        print(DASH_STRING)
        print('\n')
        for field, title, axis_lim_dict in field_title_iterator:
            print(DASH_STRING)
            print(f'{title}')
            g = sns.catplot(data=cluster_stats_per_group_df,
                            x=field,
                            y=GROUP_FIELD_NAME_STR,
                            orient='h',
                            kind='bar',
                            palette=return_color_palette(n_groups),
                            height=SEABORN_FIGURE_LEVEL_HEIGHT_WIDE_SINGLE,
                            aspect=SEABORN_FIGURE_LEVEL_ASPECT_WIDE,
                            linewidth=SEABORN_BOX_LINE_WIDTH_SINGLE,
                            facet_kws=axis_lim_dict)

            g.ax.xaxis.set_major_formatter(FuncFormatter(integer_formatter))
            g.ax.set_xlabel(title)

            plt.show(g)

    def _return_user_id_seq_id_mapping_per_group_dict(self) -> dict[int, dict[int, int]]:

        user_seq_mapping_dict_per_group = {}
        for group, df in self.sequence_distance_analytics.unique_learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR):
            user_seq_mapping_dict = {}
            for _, series in df.iterrows():
                for user_id in series[LEARNING_ACTIVITY_SEQUENCE_USERS_NAME_STR]:
                    user_seq_mapping_dict[user_id] = series[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]
            user_seq_mapping_dict_per_group[group] = user_seq_mapping_dict    
        
        return user_seq_mapping_dict_per_group

    def _return_seq_id_cluster_mapping_per_group_dict(self,
                                                      sequence_cluster_per_group_df: pd.DataFrame) -> dict[int, dict[int, tuple]]: 
        seq_id_cluster_mapping_dict_per_group = defaultdict(lambda: defaultdict(tuple))
        for group, df in sequence_cluster_per_group_df.groupby(GROUP_FIELD_NAME_STR):
            seq_id_cluster_mapping_dict = defaultdict(set)

            seq_id_cluster_iterator = zip(df[SEQUENCE_ID_FIELD_NAME_STR], df[CLUSTER_FIELD_NAME_STR])
            for seq_id, cluster in seq_id_cluster_iterator:
                seq_id_cluster_mapping_dict[seq_id].add(cluster)

            seq_id_cluster_mapping_dict = {k: tuple(v) for k,v in seq_id_cluster_mapping_dict.items()}
            seq_id_cluster_mapping_dict_per_group[group] = seq_id_cluster_mapping_dict

        return seq_id_cluster_mapping_dict_per_group

    def _return_distance_matrix(self,
                                group: int,
                                normalize_distance: bool,
                                use_unique_sequence_distances: bool):
        """Returns a sequence distance matrix for the specified group.

        Parameters
        ----------
        group : list of str, optional
            The group for which the distance matrix will be returned
        normalize_distance : bool
            A boolean indicating whether the sequence distances are being normalized between 0 and 1
        use_unique_sequence_distances: bool
            A boolean indicating whether only unique sequences are being used as the basis for distance calculations

        Returns
        -------
        dict
            A dictionary containing the group string and the corresponding sequence distance matrix
        """

        distance_matrix = self.sequence_distance_analytics.return_sequence_distance_matrix_per_group(group, 
                                                                                                     normalize_distance, 
                                                                                                     use_unique_sequence_distances)[SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR]
        return distance_matrix

    def _cluster_sequences(self,
                           group: int,
                           normalize_distance: bool,
                           use_unique_sequence_distances: bool) -> pd.DataFrame:

            distance_matrix = self._return_distance_matrix(group,
                                                           normalize_distance,
                                                           use_unique_sequence_distances)

            if self.dimensionality_reduction_param_grid_generator:
                dim_reduction_para_grid_generator = self.dimensionality_reduction_param_grid_generator(distance_matrix,
                                                                                                       ParaGridParaType.DIM_REDUCTION)
                dim_reduction_grid = dim_reduction_para_grid_generator.return_param_grid()
            else:
                dim_reduction_grid = None

            cluster_para_grid_generator = self.cluster_param_grid_generator(distance_matrix,
                                                                            ParaGridParaType.CLUSTERING)
            cluster_para_grid = cluster_para_grid_generator.return_param_grid()

            clusterer = SequenceDistanceClusters(distance_matrix,
                                                 self.dataset_name,
                                                 group,
                                                 normalize_distance,
                                                 use_unique_sequence_distances,
                                                 self.cluster_function,
                                                 self.dimensionality_reduction_function,
                                                 self.cluster_validation_functions,
                                                 cluster_para_grid,
                                                 dim_reduction_grid,
                                                 self.cluster_validation_parameters,
                                                 self.cluster_validation_attributes,
                                                 self.parallelize_computation)

            cluster_results = clusterer.return_cluster_parameter_tuning_results()

            return cluster_results

    def _return_hyperparameter_tuning_string(self,
                                             group: int,
                                             normalize_distance: bool,
                                             sequence_distance_entity: str,
                                             number_of_sequences: int,
                                             number_of_unique_sequences: int,
                                             cluster_algo_name: str,
                                             dim_reduction_algo_name: str,
                                             cluster_param_1: str,
                                             cluster_param_2: str,
                                             cluster_best_params: pd.Series,
                                             dim_reduction_best_params: pd.Series,
                                             cluster_validation_metric_name: str,
                                             cluster_validation_metric_optimum_value: float,
                                             number_clusters: int,
                                             percentage_clustered: float,
                                             seq_count_per_cluster: pd.DataFrame) -> str:

        cluster_best_params.index = [remove_param_praefix(par, ParaGridParaType.CLUSTERING) for par in cluster_best_params.index]
        dim_reduction_best_params.index = [remove_param_praefix(par, ParaGridParaType.DIM_REDUCTION) for par in dim_reduction_best_params.index]

        param_list = [cluster_param_1, cluster_param_2]
        cluster_best_params_filtered =  filter(lambda x: x[0] not in param_list, cluster_best_params.items())   

        output_string = ''
        output_string += STAR_STRING[:51]
        output_string += '\n' 
        output_string += f'{GROUP_FIELD_NAME_STR}: {group}'
        output_string += '\n' 
        output_string += STAR_STRING[:51]
        output_string += '\n\n' 
        output_string += f'{CLUSTERING_PARAMETER_TUNING_SEQ_DIST_IS_NORMALIZED_TITLE_NAME_STR}: {normalize_distance}'
        output_string += '\n' 
        output_string += f'{CLUSTERING_PARAMETER_TUNING_SEQ_DIST_ENTITIES_TITLE_NAME_STR}: {sequence_distance_entity}'
        output_string += '\n\n' 
        output_string += f'{LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR}: {number_of_sequences}'
        output_string += '\n' 
        output_string += f'{LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR}: {number_of_unique_sequences}'
        output_string += '\n\n'
        output_string += DASH_STRING[:51]
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_DIM_REDUCTION_ALGORITHM_TITLE_NAME_STR}{dim_reduction_algo_name}'
        output_string += '\n\n'
        output_string += CLUSTERING_PARAMETER_TUNING_FIXED_PARAMS_NAME_STR
        output_string += '\n'
        for field, value in dim_reduction_best_params.items():
            output_string += f'     - {field}: {value}'
            output_string += '\n'
        output_string += '\n'
        output_string += DASH_STRING[:51]
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_ALGORITHM_TITLE_NAME_STR}{cluster_algo_name}'
        output_string += '\n\n'
        output_string += CLUSTERING_PARAMETER_TUNING_VARYING_PARAMS_NAME_STR
        output_string += '\n'
        output_string += f'     - {cluster_param_1}'
        output_string += '\n'
        output_string += f'     - {cluster_param_2}'
        output_string += '\n\n'
        output_string += CLUSTERING_PARAMETER_TUNING_FIXED_PARAMS_NAME_STR
        output_string += '\n'
        for field, value in cluster_best_params_filtered:
            output_string += f'     - {field}: {value}'
            output_string += '\n'
        output_string += '\n'
        output_string += DASH_STRING[:51]
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_TITLE_NAME_STR}{cluster_validation_metric_name}'
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_VALIDATION_METRIC_OPTIMUM_VALUE_TITLE_NAME_STR}{cluster_validation_metric_optimum_value:.3f}'
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_VALIDATION_PERCENTAGE_CLUSTERED_TITLE_NAME_STR}{percentage_clustered:.3f} %'
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_VALIDATION_NUMBER_CLUSTERS_TITLE_NAME_STR}{number_clusters}'
        output_string += '\n\n'
        output_string += DASH_STRING[:51]
        output_string += '\n\n'
        output_string += f'{CLUSTERING_PARAMETER_TUNING_VALIDATION_NUMBER_SEQUENCES_PER_CLUSTER_TITLE_NAME_STR}'
        output_string += '\n\n'
        output_string += f'{seq_count_per_cluster.to_string(index=True)}'
        output_string += '\n\n'
        output_string += STAR_STRING[:51]

        return output_string
    
    def _select_best_cluster_results_embedding(self,
                                               cluster_results_per_group: pd.DataFrame,
                                               cluster_validation_metric: str,
                                               cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        cluster_results_per_group = cluster_results_per_group.sort_values(by=cluster_validation_metric,
                                                                          ascending=cluster_validation_lower_is_better)

        dim_reduction_param_series = cluster_results_per_group[self.dim_reduct_para_fields].iloc[0, :]

        # only select best dim reduction results
        mask = pd.Series([True] * cluster_results_per_group.shape[0])
        for field, value in dim_reduction_param_series.items():
            mask &= (cluster_results_per_group[field] == value)

        cluster_results_per_group = cluster_results_per_group.loc[mask, :]
    
        return cluster_results_per_group

    def _calc_hyperparameter_tuning_matrix_plotting_data(self,
                                                         cluster_param_1: str,
                                                         cluster_param_2: str,
                                                         cluster_validation_metric: str,
                                                         cluster_validation_lower_is_better: bool) -> tuple[pd.DataFrame, 
                                                                                                            dict[int, AlgoParams], 
                                                                                                            dict[int, ClusterValidationResults], 
                                                                                                            dict[int, HeatmapAnnotationVars]]:

        data_long_list = []
        algo_params_dict = {}
        cluster_validation_results_dict = {}
        heatmap_annotation_dict = {}

        for group, data in self.cluster_results_per_group.groupby(GROUP_FIELD_NAME_STR):

            data = self._select_best_cluster_results_embedding(data,
                                                               cluster_validation_metric,
                                                               cluster_validation_lower_is_better)

            cluster_algo_name = self.cluster_algo_name
            dim_reduction_algo_name = self.dimensionality_reduction_algo_name
            cluster_param_series = data[self.cluster_para_fields].iloc[0, :]
            dim_reduction_param_series = data[self.dim_reduct_para_fields].iloc[0, :]

            cluster_validation_metric_value = data[cluster_validation_metric].iloc[0]
            number_clusters = data[CLUSTERING_NUMBER_CLUSTERS_NAME_STR].iloc[0]
            percentage_clustered = data[CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR].iloc[0]

            algo_params_dict[group] = AlgoParams(cluster_algo_name,
                                                 dim_reduction_algo_name,
                                                 cluster_param_series,
                                                 dim_reduction_param_series)

            cluster_validation_results_dict[group] = ClusterValidationResults(cluster_validation_metric,
                                                                              cluster_validation_metric_value,
                                                                              number_clusters,
                                                                              percentage_clustered)

            # create data in long format for heatmap in facet grid
            data_long = pd.melt(data,
                                id_vars=[GROUP_FIELD_NAME_STR, cluster_param_1, cluster_param_2],
                                value_vars=[cluster_validation_metric, CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR, CLUSTERING_NUMBER_CLUSTERS_NAME_STR],
                                var_name=CLUSTERING_PARAMETER_TUNING_RESULT_METRIC_NAME_STR,
                                value_name=CLUSTERING_PARAMETER_TUNING_RESULT_VALUE_NAME_STR).sort_values(by=[cluster_param_1, cluster_param_2])
            data_long = data_long.reset_index(drop=True)
            data_long_list.append(data_long)

            # create heatmap annotation data
            data_long = data_long.loc[data_long[CLUSTERING_PARAMETER_TUNING_RESULT_METRIC_NAME_STR]==cluster_validation_metric].copy()
            data_matrix = data_long.pivot(index=cluster_param_2, 
                                          columns=cluster_param_1, 
                                          values=CLUSTERING_PARAMETER_TUNING_RESULT_VALUE_NAME_STR)
            data_matrix = data_matrix.sort_index(axis=0, ascending=False).sort_index(axis=1)

            if cluster_validation_lower_is_better:
                optimum_value = np.min(data_matrix.values)
            else:
                optimum_value = np.max(data_matrix.values)
            optimum_values_idx = np.argwhere(data_matrix.values.flatten() == optimum_value).flatten()

            # Create a masked array
            masked_data_matrix = np.ma.masked_where(data_matrix != optimum_value, data_matrix)

            masked_data_matrix_df = pd.DataFrame(masked_data_matrix)
            masked_data_matrix_df.columns =  data_matrix.columns
            masked_data_matrix_df.index =  data_matrix.index
            masked_data_matrix_df.columns.name = data_matrix.columns.name
            masked_data_matrix_df.index.name = data_matrix.index.name

            optimum_value_positions = [(i, j) for i in range(data_matrix.shape[0]) for j in range(data_matrix.shape[1]) if data_matrix.values[i, j] == optimum_value]

            heatmap_annotation_dict[group] = HeatmapAnnotationVars(masked_data_matrix_df,
                                                                   masked_data_matrix,
                                                                   optimum_value_positions,
                                                                   optimum_values_idx)

        data_long_all_group = pd.concat(data_long_list) 

        return data_long_all_group, algo_params_dict, cluster_validation_results_dict, heatmap_annotation_dict

    def _calc_cluster_per_group_plotting_data(self,
                                              group: int,
                                              cluster_results: ClusterResults,
                                              reducer: Callable,
                                              use_best_dimension_reduction_params: bool) -> pd.DataFrame:

        distance_matrix = self._return_distance_matrix(group,
                                                       self.normalize_distance,
                                                       self.use_unique_sequence_distances)
        if use_best_dimension_reduction_params:
            params_dict = cluster_results.best_dim_reduction_parameters
            params_dict_praefix_removed = ({remove_param_praefix(k, ParaGridParaType.DIM_REDUCTION): v 
                                                                for k, v in params_dict.items()})
            _ = params_dict_praefix_removed.pop(CLUSTERING_N_COMPONENTS_NAME_STR)
        else:
            params_dict_praefix_removed = {}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            dim_reducer = reducer(n_components=2,
                                  **params_dict_praefix_removed)

            embedding_2D = dim_reducer.fit_transform(distance_matrix.values)

        clustered_labels = cluster_results.cluster_labels[cluster_results.clustered]

        emb_clustered = embedding_2D[cluster_results.clustered, :]
        emb_not_clustered = embedding_2D[~cluster_results.clustered, :]
        
        emb_clustered_df = pd.DataFrame(emb_clustered, columns=[CLUSTERING_2D_PLOT_CLUSTERED_X_AXIS_FIELD_NAME_STR, CLUSTERING_2D_PLOT_CLUSTERED_Y_AXIS_FIELD_NAME_STR])
        emb_clustered_df[CLUSTER_FIELD_NAME_STR] = clustered_labels
        emb_not_clustered_df = pd.DataFrame(emb_not_clustered, columns=[CLUSTERING_2D_PLOT_NOT_CLUSTERED_X_AXIS_FIELD_NAME_STR, CLUSTERING_2D_PLOT_NOT_CLUSTERED_Y_AXIS_FIELD_NAME_STR])
        emb_df = pd.concat([emb_clustered_df, emb_not_clustered_df], ignore_index=False, axis=1)
        emb_df[GROUP_FIELD_NAME_STR] = group

        return emb_df

    def _calc_cluster_stats_per_group_plotting_data(self,
                                                    cluster_validation_metric: str,
                                                    cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        cluster_results = self._return_cluster_results_per_group_dict(cluster_validation_metric,
                                                                                cluster_validation_lower_is_better)

        group_array = []
        number_clusters_array = []
        percentage_clustered_array = []
        min_cluster_size_array = []
        sequence_distances_is_normalized_array = []
        cluster_entity_type_array = []

        for group, cluster_results in cluster_results.items():

            group_array.append(group)
            number_clusters_array.append(cluster_results.number_clusters)
            percentage_clustered_array.append(cluster_results.percentage_clustered)
            min_cluster_size_array.append(cluster_results.min_cluster_size)
            sequence_distances_is_normalized_array.append(cluster_results.sequence_distances_is_normalized)
            cluster_entity_type_array.append(cluster_results.cluster_entity_type)

        cluster_stats_per_group_df = pd.DataFrame({DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
                                                   GROUP_FIELD_NAME_STR: group_array,
                                                   CLUSTERING_NUMBER_CLUSTERS_NAME_STR: number_clusters_array,
                                                   CLUSTERING_PERCENTAGE_CLUSTERED_NAME_STR: percentage_clustered_array,
                                                   CLUSTERING_SMALLEST_CLUSTER_SIZE_NAME_STR: min_cluster_size_array,
                                                   CLUSTERING_CLUSTER_SEQ_DIST_IS_NORMALIZED_NAME_STR: sequence_distances_is_normalized_array,
                                                   CLUSTERING_CLUSTER_SEQ_DIST_ENTITY_NAME_STR: cluster_entity_type_array})

        return cluster_stats_per_group_df

    def _return_cluster_results_per_group_dict(self,
                                               cluster_validation_metric: str,
                                               cluster_validation_lower_is_better: bool) -> dict[int, ClusterResults]:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        cluster_results_dict = {}
        for group, data in self.cluster_results_per_group.groupby(GROUP_FIELD_NAME_STR):

            data = self._select_best_cluster_results_embedding(data,
                                                               cluster_validation_metric,
                                                               cluster_validation_lower_is_better)
            cluster_result_series = data.iloc[0, :][self._cluster_results_list]
            cluster_results = list(cluster_result_series.values)
            best_dim_reduction_parameters = data.iloc[0, :][self.dim_reduct_para_fields].to_dict()
            cluster_results.append(best_dim_reduction_parameters)

            cluster_results_dict[group] = ClusterResults(*cluster_results)
        
        return cluster_results_dict
    
    def _return_best_cluster_results_per_group_df(self,
                                                  cluster_validation_metric: str,
                                                  cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        cluster_results_per_group_df = (self.cluster_results_per_group.sort_values(by=cluster_validation_metric,
                                                                                   ascending=cluster_validation_lower_is_better)
                                                                      .groupby(GROUP_FIELD_NAME_STR)
                                                                      .first()
                                                                      .reset_index())
        field_list = cluster_results_per_group_df.columns.to_list()
        idx = field_list.index(DATASET_NAME_FIELD_NAME_STR)
        dataset_name = field_list.pop(idx)
        field_list.insert(0, dataset_name)

        cluster_results_per_group_df = cluster_results_per_group_df[field_list]

        return cluster_results_per_group_df

    def _return_sequence_cluster_per_group_df(self,
                                              cluster_validation_metric: str,
                                              cluster_validation_lower_is_better: bool) -> pd.DataFrame:

        cluster_results = self._return_cluster_results_per_group_dict(cluster_validation_metric,
                                                                      cluster_validation_lower_is_better)
        group_array = []
        cluster_id_array = []
        seq_id_array = []
        user_id_array = []
        for group, cluster_results in cluster_results.items():
            array_len = len(cluster_results.cluster_entity_ids)
            group_ids = [group] * array_len
            if self.use_unique_sequence_distances:
                seq_ids = cluster_results.cluster_entity_ids
                user_ids = [None] * array_len
            else:
                seq_ids = tuple([self.user_seq_mapping_dict_per_group[group][cluster_entity] for cluster_entity in cluster_results.cluster_entity_ids])
                user_ids = cluster_results.cluster_entity_ids

            group_array.extend(group_ids)
            cluster_id_array.extend(cluster_results.cluster_labels)
            seq_id_array.extend(seq_ids)
            user_id_array.extend(user_ids)

        sequence_cluster_per_group_df = pd.DataFrame({DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
                                                      GROUP_FIELD_NAME_STR: group_array,
                                                      CLUSTER_FIELD_NAME_STR: cluster_id_array,
                                                      SEQUENCE_ID_FIELD_NAME_STR: seq_id_array,
                                                      USER_FIELD_NAME_STR: user_id_array})
        
        return sequence_cluster_per_group_df
    
    def _add_cluster_id_to_sequence_stats_df(self,
                                             sequence_stats_df: pd.DataFrame,
                                             sequence_cluster_per_group_df: pd.DataFrame) -> pd.DataFrame:

        sequence_stats_df = sequence_stats_df.copy()
        seq_id_cluster_mapping_per_group_dict = self._return_seq_id_cluster_mapping_per_group_dict(sequence_cluster_per_group_df)

        # helper function
        def mapping_func(row):
            return seq_id_cluster_mapping_per_group_dict[row[GROUP_FIELD_NAME_STR]][row[SEQUENCE_ID_FIELD_NAME_STR]]

        cluster_series = sequence_stats_df.apply(mapping_func, axis=1)
        sequence_stats_df[CLUSTER_FIELD_NAME_STR] = cluster_series

        return sequence_stats_df

    def _return_n_sequences_per_cluster_per_group_df(self,
                                                     sequence_cluster_per_group_df: pd.DataFrame) -> pd.DataFrame:

        n_seq_per_cluster_df = (sequence_cluster_per_group_df.groupby([GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR])[SEQUENCE_ID_FIELD_NAME_STR]
                                                             .agg([len, pd.Series.nunique]))
        n_seq_per_cluster_df.columns = [CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR,
                                        CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR]
        n_seq_per_cluster_df = n_seq_per_cluster_df.reset_index()

        return n_seq_per_cluster_df

    def _return_n_sequences_per_cluster_per_group_dict(self,
                                                       cluster_validation_metric: str,
                                                       cluster_validation_lower_is_better: bool) -> dict[int, pd.DataFrame]:

        _ = check_value_not_none(self.cluster_results_per_group,
                                 CLUSTERING_PARAMETER_TUNING_ERROR_NO_RESULTS_NAME_STR)

        _ = check_value_in_iterable(cluster_validation_metric,
                                    self.cluster_validation_algo_names)

        sequence_cluster_per_group_df = self.return_sequence_cluster_per_group_df(cluster_validation_metric,
                                                                                  cluster_validation_lower_is_better)

        n_seq_per_cluster_df = self._return_n_sequences_per_cluster_per_group_df(sequence_cluster_per_group_df)
        n_seq_per_cluster_df.columns = [GROUP_FIELD_NAME_STR, 
                                        CLUSTER_FIELD_NAME_STR, 
                                        CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_SHORT_FIELD_NAME_STR, 
                                        CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_SHORT_FIELD_NAME_STR]

        n_sequences_per_cluster_per_group_dict = {group: df.drop(GROUP_FIELD_NAME_STR, axis=1).set_index(CLUSTER_FIELD_NAME_STR) 
                                                  for group, df in n_seq_per_cluster_df.groupby(GROUP_FIELD_NAME_STR)}
        
        return n_sequences_per_cluster_per_group_dict

    def _return_object_name(self,
                            obj: Any) -> str | None:

        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = None
        
        return name

# class ClusterAnalysisPlotGroupSelectionCriteria():
#     """A class used for plotting the selections criteria(min_sequence_number_per_group_threshold, 
#     min_unique_sequence_number_per_group_threshold, mean_sequence_distance_range) for groups to be included
#     in the cluster analysis.
    
#     Parameters
#     ----------
#     dataset_name: str
#         The name of the dataset.
#     learning_activity_sequence_stats_per_group : pd.DataFrame
#         A learning activity sequence statistics per group dataframe
#     sequence_distances_dict: dict
#         A nested dictionary containing results of "calculate_sequence_distances"\
#         For each group the subdictionary must contain the following keys: 
#         ('Sequence Distance', 'Sequence Maximum Length', 'Sequence User Combination', 'User', 'Sequence Length', 'Sequence ID', 'Sequence Array').
#     use_normalized_sequence_distance: bool
#         A flag indicating whether a normalized sequence distance, ranging from 0 to 1, will be used for clustering.
#     min_sequence_number_per_group_threshold: int
#         The number of sequences a group must have to be included in the cluster analysis
#     min_unique_sequence_number_per_group_threshold: int
#         The number of unique sequences a group must have to be included in the cluster analysis
#     mean_sequence_distance_range: tuple
#         The mean sequence distance lower and upper bounds a group is allowed to have to be included in the cluster
#         analysis

#     Methods
#     -------
#     display_group_selection_criteria   
#         Generates 2 plots:
#         1.) Displays the relationship between unique and total number of sequences per group and marks the respective specified
#         thresholds required for a group to be included in the cluster analysis
#         2.) Displays the sequence distances per group and marks the respective specified
#         thresholds required for a group to be included in the cluster analysis

#     Attributes
#     -------
#     count_df
#         A dataframe containing data to plot the relationship between unique and total number of sequences per group
#     seq_dist_per_group_df
#         A dataframe containing data to plot the distribution of sequence distances per group
#     """
#     def __init__(self,
#                  dataset_name: str,
#                  learning_activity_sequence_stats_per_group: pd.DataFrame,
#                  seq_distances,
#                  use_normalized_sequence_distance: bool,
#                  min_sequence_number_per_group_threshold: int,
#                  min_unique_sequence_number_per_group_threshold: int,
#                  mean_sequence_distance_range: tuple):

#         self.dataset_name = dataset_name
#         self.learning_activity_sequence_stats_per_group = learning_activity_sequence_stats_per_group
#         self.seq_distances = seq_distances
#         self.use_normalized_sequence_distance = use_normalized_sequence_distance
#         self.min_sequence_number_per_group_threshold = min_sequence_number_per_group_threshold
#         self.min_unique_sequence_number_per_group_threshold = min_unique_sequence_number_per_group_threshold
#         self.mean_sequence_distance_range = mean_sequence_distance_range
        
#     def display_group_selection_criteria(self):

#         # all groups in one figure - unique seq count vs seq count -> generate df for plotting
#         self.count_df = self.learning_activity_sequence_stats_per_group.groupby(GROUP_FIELD_NAME_STR).head(1)
#         self.count_df = self.count_df.copy()
#         self.count_df[CLUSTERING_DATASET_NAME_FIELD_NAME_STR] = self.dataset_name
#         ylim = self.count_df[LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR].max()

#         # sequence distances per group -> generate df for plotting
#         group_list = []
#         distances_list = []
#         normalized_distances_list = []
#         max_sequence_length_list = []
#         for group, subdict in self.seq_distances.items():

#             # extract data from dictionary
#             distances = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
#             max_sequence_len_per_distance = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
#             # choose between normalized and non-normalized sequence distance
#             if self.use_normalized_sequence_distance:
#                 distance_array = distances / max_sequence_len_per_distance 
#             else:
#                 distance_array = distances

#             group_list.extend([group]*len(distances))
#             distances_list.extend(distance_array)

#         seq_dist_per_group_dict = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
#                                    GROUP_FIELD_NAME_STR: group_list,
#                                    LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR: distances_list}

#         self.seq_dist_per_group_df = pd.DataFrame(seq_dist_per_group_dict)

#         # plot all groups in one figure - unique seq count vs seq count
#         print(STAR_STRING)
#         print(STAR_STRING)
#         print(' ')
#         print(DASH_STRING)
#         print(f'{CLUSTER_FIELD_NAME_STR} Analysis - {GROUP_FIELD_NAME_STR} Selection Criteria')
#         print(DASH_STRING)
#         print(' ')
#         print(f'{LEARNING_ACTIVITY_UNIQUE_VS_TOTAL_NUMBER_OF_SEQUENCES_PER_GROUP_TITLE_NAME_STR}:')
#         g = sns.scatterplot(data=self.count_df,
#                             x=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
#                             y=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
#                             s=100, 
#                             alpha=0.7)
#         g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_COUNT_PER_GROUP_NAME_STR, 
#             ylabel=LEARNING_ACTIVITY_UNIQUE_SEQUENCE_COUNT_PER_GROUP_NAME_STR,
#             ylim=(-5,ylim))
#         g.axline(xy1=(0,0), slope=1, color='r', linewidth=3);

#         if self.min_sequence_number_per_group_threshold:
#             g.axvline(x=self.min_sequence_number_per_group_threshold, ymin=0, ymax=1, color='orange', linewidth=5);
#         if self.min_unique_sequence_number_per_group_threshold:
#             g.axhline(y=self.min_unique_sequence_number_per_group_threshold, xmin=0, xmax=1, color='orange', linewidth=5);
#         plt.show(g)

#         # plot sequence distance per group
#         if self.use_normalized_sequence_distance:
#             print(f'{LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#         else:
#             print(f'{LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#         print(f'Base: All {USER_FIELD_NAME_STR}-{SEQUENCE_STR} Combinations')
#         g = sns.boxplot(data=self.seq_dist_per_group_df, 
#                         x=LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR, 
#                         y=GROUP_FIELD_NAME_STR,
#                         showmeans=True, 
#                         meanprops=marker_config);

#         for patch in g.patches:
#             r, g, b, a = patch.get_facecolor()
#             patch.set_facecolor((r, g, b, 0.5))

#         g = sns.stripplot(data=self.seq_dist_per_group_df, 
#                         x=LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR, 
#                         y=GROUP_FIELD_NAME_STR,
#                         size=2, 
#                         color="red",
#                         alpha=0.1)
#         if self.min_sequence_number_per_group_threshold:
#             g.axvline(x=self.mean_sequence_distance_range[0], ymin=0, ymax=1, color='orange', linewidth=5);
#             g.axvline(x=self.mean_sequence_distance_range[1], ymin=0, ymax=1, color='orange', linewidth=5);
#         if self.use_normalized_sequence_distance:
#             g.set(xlabel=LEARNING_ACTIVITY_NORMALIZED_SEQUENCE_DISTANCE_NAME_STR);
#         else:
#             g.set(xlabel=LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR);
#         plt.show()
#         print(' ')
#         print(STAR_STRING)
#         print(STAR_STRING)

# class ClusterEvaluation:
#     """
#     A class containing methods which perform several tasks:
#         1. Calculation of sequence distance clusters for each group

#         2. Performance of omnibus testing for central tendency differences in a specified evaluation metric
#            between each cluster
#             - Depending on normality and homoscedasticity of the evaluation metric in the clusters an appropriate test
#               will be chosen: Anova vs Welch-Anova vs Kruskall-Wallis
        
#         3. Performance of post-hoc tests for differences between single clusters of a group

#         4. Calculation of sequence and cluster statistics for each group

#         5. Plotting of cluster results and differences in distribution of specified evaluation metric between clusters

#     Parameters
#     ----------
#     dataset_name: str
#         The name of the dataset.
#     interactions: pd.DataFrame
#         The interactions dataframe.
#     user_field: str
#         Then name of the user field.
#     group_field: str
#         Then name of the group field.
#         This argument should be set to None if the interactions dataframe does not have a group_field.
#     evaluation_field: str
#         Then name of the evaluation metric field for which central tendency differences between clusters are being tested.
#     sequence_distances_dict: dict
#         A nested dictionary containing results of "calculate_sequence_distances"
#         For each group the subdictionary must contain the following keys: 
#         ('Sequence Distance', 'Sequence Maximum Length', 'Sequence User Combination', 'User', 'Sequence Length', 'Sequence ID', 'Sequence Array').
#     use_normalized_sequence_distance: bool
#         A flag indicating whether a normalized sequence distance, ranging from 0 to 1, will be used for clustering.
#     min_sequence_number_per_group_threshold: int
#         The number of sequences a group must have to be included in the cluster analysis
#     min_unique_sequence_number_per_group_threshold: int
#         The number of uniuqe sequences a group must have to be included in the cluster analysis
#     mean_sequence_distance_range: tuple
#         The mean sequence distance lower and upper bounds a group is allowed to have to be included in the cluster
#         analysis
#     cluster_function
#         A scikit-learn compatible (syntax- and method-wise) cluster algorithm.
#     hdbscan_min_cluster_size_as_pct_of_group_seq_num: float, None
#         The minimal cluster size as percentage of number of sequences per group. This parameters only applies if\
#         cluster_function is equal to the HDBSCAN class of the hdbscan library and otherwise should be set to
#         None.
#     normality_test_alpha: float
#         A significance level for the normality test (shapiro) -> used for testing anova assumptions.
#     homoscedasticity_test_alpha: float
#         A significance level for the homoscedasticity test (levene) -> used for testing anova assumptions.
#     **kwargs
#         Keyword arguments functioning as parameters for the specified cluster_function.

#     Methods
#     -------
#     cluster_sequences_and_test_eval_metric_diff   
#         Clusters the user sequence distances for each group and performs omnibus test for central tendency differences between the clusters.

#     perform_post_hoc_pairwise_eval_metric_diff_tests(group_str: str
#                                                      alternative: str)
#         Performs pairwise post-hoc tests for differences in central tendency of the evaluation metric between clusters.
        
#     aggregate_sequence_clustering_and_eval_metric_diff_test_over_groups
#         Aggregates the clustering results over all groups and inter alia calculates the percentage of groups having\
#         significant differences in central tendency of the evaluation metric between clusters.

#     display_included_groups
#         Displays the absolute value and percentages of groups included in the cluster analysis.

#     display_clusters_by_group_umap(group_str: str
#                                    **kwargs)
#         Reduces the sequence distance matrix of specified group with UMAP and displays clustering in a two-dimensional projection.

#     display_clusters_by_group_pca(group_str: str)
#         Reduces the sequence distance matrix of specified group with PCA and displays clustering in a two-dimensional projection.
    
#     display_eval_metric_dist_between_clusters_by_group(group_str: str)
#         Displays the evaluation metric distribution per cluster via boxplots, violinplot, boxenplot and kde for specified group.
    
#     display_cluster_size_by_group(group_str: str)
#         Displays the cluster sizes for specified group.
    
#     display_eval_metric_dist_between_cluster_all_groups
#         Displays the evaluation metric distribution per cluster via boxplots for all groups.

#     display_clusters_all_group_umap
#         Reduces the sequence distance matrices of all group with UMAP and displays clustering in a two-dimensional projection.

#     display_clusters_all_group_pca
#         Reduces the sequence distance matrices of all group with PCA and displays clustering in a two-dimensional projection.

#     display_number_of_clusters_all_group
#         Displays the number of clusters per group as barplot.

#     display_min_cluster_size_all_group:
#         Displays the size of the smallest cluster per group as barplot.
    
#     display_percentage_clustered_all_group
#         Displays the percentage of sequencese clustered per group as barplot.

#     display_number_sequences_per_cluster_all_group
#         Displays the number of sequences per cluster for each group as multi-grid barplot.
    
#     display_number_unique_sequences_per_cluster_all_group
#         Displays the number of unique sequences per cluster for each group as multi-grid barplot.

#     display_number_unique_sequences_vs_number_sequences_per_cluster_all_group
#         Displays the relationship between the number of unique sequences vs the number of sequences per cluster for each group as multi-grid scatterplot.

#     print_number_of_clusters_all_group
#         Print the number of clusters for each group.

#     print_min_cluster_sizes_all_group
#         Print the minimum cluster sizes sizes for each group.

#     print_percentage_clustered_all_group
#         Print the percentage of sequences clustered for each group.

#     Attributes
#     -------
#     cluster_eval_metric_central_tendency_differences_per_group
#         A dataframe containing groupwise sequence and cluster stats, assumptions(normality and homoscedasticity) test results for the omnibus test and
#         and omnibus central tendency differences of evaluation metrics between clusters test results

#     cluster_stats_per_group
#         A dataframe containing cluster information for each group

#     aggregate_sequence_clustering_and_eval_metric_diff_test
#         A dataframe containing aggregates of cluster_eval_metric_central_tendency_differences_per_group. Contains the main result
#         field (percentage of groups with significant central tendency differences in evaluation metric) and includes number of sequence,
#         number of unique sequence and number of cluster stats per group.
    
#     group_cluster_analysis_inclusion_status
#         A dataframe containing information with respect to which group satisfies the conditions required to be included
#         the cluster analysis (specified via the parameters min_sequence_number_per_group_threshold, 
#         min_unique_sequence_number_per_group_threshold, mean_sequence_distance_range).
    
#     min_cluster_size_correction_df
#         A dataframe containing information for which group the min_cluster_size parameter needed to be adjusted to a value
#         of 2 due to being smaller than 2. Will only be calculated if hdbscan_min_cluster_size_as_pct_of_group_seq_num
#         is not None.
#     """
#     def __init__(self, 
#                  dataset_name: str,
#                  interactions: pd.DataFrame,
#                  user_field: str,
#                  group_field: str,
#                  evaluation_field: str,
#                  sequence_distances_dict: dict,
#                  use_normalized_sequence_distance: bool,
#                  min_sequence_number_per_group_threshold: int,
#                  min_unique_sequence_number_per_group_threshold: int,
#                  mean_sequence_distance_range: tuple,
#                  cluster_function,
#                  hdbscan_min_cluster_size_as_pct_of_group_seq_num: float,
#                  normality_test_alpha: float,
#                  homoscedasticity_test_alpha: float,
#                  **kwargs):

#         self.dataset_name = dataset_name
#         self.interactions = interactions
#         self.user_field = user_field
#         self.group_field = group_field
#         self.evaluation_field = evaluation_field
#         self.sequence_distances_dict= sequence_distances_dict
#         self.use_normalized_sequence_distance = use_normalized_sequence_distance
#         self.min_sequence_number_per_group_threshold = min_sequence_number_per_group_threshold
#         self.min_unique_sequence_number_per_group_threshold = min_unique_sequence_number_per_group_threshold
#         self.mean_sequence_distance_range = mean_sequence_distance_range
#         self.cluster_function = cluster_function
#         self.hdbscan_min_cluster_size_as_pct_of_group_seq_num = hdbscan_min_cluster_size_as_pct_of_group_seq_num
#         self.normality_test_alpha = normality_test_alpha 
#         self.homoscedasticity_test_alpha = homoscedasticity_test_alpha
#         self.kwargs = kwargs

#         # flag indicating whether data is already clustered
#         self._data_clustered = False

#         # the central results of ClusterEvaluation
#         self.cluster_eval_metric_central_tendency_differences_per_group = None
#         self.cluster_stats_per_group = None
#         self.aggregate_sequence_clustering_and_eval_metric_diff_test = None
#         self.group_cluster_analysis_inclusion_status = None
#         self.min_cluster_size_correction_df = None
        
#         # intermediate results
#         self._square_matrix_per_group = {}
#         self._user_cluster_mapping_per_group = {}
#         self._user_sequence_id_mapping_per_group = {}
#         self._user_sequence_array_mapping_per_group = {}
#         self._user_sequence_length_mapping_per_group = {}
#         self._clustered_per_group = {}
#         self._percentage_clustered_per_group = {}
#         self._user_cluster_eval_metric_df_per_group = {}

#     def cluster_sequences_and_test_eval_metric_diff(self):
#         """Clusters the user sequence distances for each group and performs omnibus test for central tendecy differences between the clusters.
#         """        
#         # initialization of result dictionaries used for generating result dataframes
#         if self.use_normalized_sequence_distance:
#             results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
#                                       CLUSTERING_GROUP_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MIN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MAX_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_STD_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_IQR_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_ALGORITHM_FIELD_NAME_STR:[],
#                                       CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR:[],
#                                       CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR:[],
#                                       CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR:[]}
#         else:
#             results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
#                                       CLUSTERING_GROUP_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEAN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEDIAN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MIN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MAX_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_STD_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_IQR_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
#                                       CLUSTERING_ALGORITHM_FIELD_NAME_STR:[],
#                                       CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR:[],
#                                       CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR:[],
#                                       CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR:[],
#                                       CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR:[],
#                                       CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR:[]}
        

#         cluster_stats_results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
#                                                 CLUSTERING_GROUP_FIELD_NAME_STR:[],
#                                                 CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
#                                                 CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
#                                                 CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
#                                                 CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR:[],
#                                                 LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR:[],
#                                                 LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR:[],
#                                                 LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR:[],
#                                                 CLUSTERING_MEAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_MEDIAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_MIN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_MAX_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_STD_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
#                                                 CLUSTERING_IQR_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[]}

#         group_cluster_analysis_inclusion_status_dict = {DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
#                                                         GROUP_FIELD_NAME_STR: [],
#                                                         CLUSTERING_GROUP_INCLUDED_IN_CLUSTER_ANALYSIS_NAME_STR: [],
#                                                         CLUSTERING_MIN_SEQUENCE_NUMBER_VALUE_NAME_STR: [],
#                                                         CLUSTERING_VIOLATED_MIN_SEQUENCE_NUMBER_NAME_STR: [],
#                                                         CLUSTERING_MIN_UNIQUE_SEQUENCE_NUMBER_VALUE_NAME_STR: [], 
#                                                         CLUSTERING_VIOLATED_MIN_UNIQUE_SEQUENCE_NUMBER_NAME_STR: [],
#                                                         CLUSTERING_MEAN_SEQUENCE_DISTANCE_RANGE_VALUE_NAME_STR: [],
#                                                         CLUSTERING_VIOLATED_MEAN_SEQUENCE_DISTANCE_RANGE_NAME_STR: []}
        
#         # if a different cluster algorithm to HDBSCAN is used, delete key for chosen minimal cluster size per group
#         if not self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
#             _ = results_dict_per_group.pop(CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR)

#         # a dictionary containing information whether the min_cluster_size of hdbscan needed to be corrected to 2
#         if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
#             min_cluster_size_correction_dict = {DATASET_NAME_FIELD_NAME_STR: self.dataset_name,
#                                                 GROUP_FIELD_NAME_STR: [],
#                                                 CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_IS_CORRECTED_NAME_STR: [],
#                                                 CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_UNCORRECTED_NAME_STR: [],
#                                                 CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_CORRECTED_NAME_STR: []}

#         # loop over all groups in the sequence distance dictionary
#         for group, subdict in tqdm(self.sequence_distances_dict.items()):

#             # extract data from dictionary
#             users = subdict[LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR]
#             distances = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
#             max_sequence_len_per_distance = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
#             sequence_ids = subdict[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]
#             sequence_arrays = subdict[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR]
#             sequence_lengths = subdict[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]
#             number_of_sequences = len(sequence_ids)
#             number_of_unique_sequences = len(np.unique(sequence_ids))

#             # choose between normalized and non-normalized sequence distance
#             if self.use_normalized_sequence_distance:
#                 distance_array = distances / max_sequence_len_per_distance 
#             else:
#                 distance_array = distances

#             # check if group fulfills the conditions to be included in the cluster analysis
#             is_included_in_cluster_analysis = True
#             min_sequence_number_violated = False
#             min_unique_sequence_number_violated = False
#             mean_sequence_distance_range_violated = False
#             if self.min_sequence_number_per_group_threshold:
#                 if len(sequence_ids) < self.min_sequence_number_per_group_threshold:
#                     is_included_in_cluster_analysis = False
#                     min_sequence_number_violated = True

#             if self.min_unique_sequence_number_per_group_threshold:
#                 if len(np.unique(sequence_ids)) < self.min_unique_sequence_number_per_group_threshold:
#                     is_included_in_cluster_analysis = False
#                     min_unique_sequence_number_violated = True

#             if self.mean_sequence_distance_range:
#                 if (np.mean(distance_array) < self.mean_sequence_distance_range[0]) or (np.mean(distance_array) > self.mean_sequence_distance_range[1]):
#                     is_included_in_cluster_analysis = False
#                     mean_sequence_distance_range_violated = True

#             group_cluster_analysis_inclusion_status_dict[GROUP_FIELD_NAME_STR].append(group)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_GROUP_INCLUDED_IN_CLUSTER_ANALYSIS_NAME_STR].append(is_included_in_cluster_analysis)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_MIN_SEQUENCE_NUMBER_VALUE_NAME_STR].append(self.min_sequence_number_per_group_threshold)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_VIOLATED_MIN_SEQUENCE_NUMBER_NAME_STR].append(min_sequence_number_violated)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_MIN_UNIQUE_SEQUENCE_NUMBER_VALUE_NAME_STR].append(self.min_unique_sequence_number_per_group_threshold)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_VIOLATED_MIN_UNIQUE_SEQUENCE_NUMBER_NAME_STR].append(min_unique_sequence_number_violated)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_MEAN_SEQUENCE_DISTANCE_RANGE_VALUE_NAME_STR].append(self.mean_sequence_distance_range)
#             group_cluster_analysis_inclusion_status_dict[CLUSTERING_VIOLATED_MEAN_SEQUENCE_DISTANCE_RANGE_NAME_STR].append(mean_sequence_distance_range_violated)

#             # skip group if it does not fullfil the conditions to be included in the cluster analysis
#             if not is_included_in_cluster_analysis:
#                 continue
            
#             # sequence distance stats per group
#             mean_sequence_metric = np.mean(distance_array)
#             median_sequence_metric = np.median(distance_array)
#             min_sequence_metric = np.min(distance_array)
#             max_sequence_metric = np.max(distance_array)
#             std_sequence_metric = np.std(distance_array)
#             iqr_sequence_metric = iqr(distance_array)

#             # generate distance matrix used for clustering:
#             square_matrix = squareform(distance_array)
#             self._square_matrix_per_group[group] = square_matrix

#             # initialize clusterer object, cluster data and calculate cluster labels for each user sequence
#             if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
#                 hdbscan_min_cluster_size = round(self.hdbscan_min_cluster_size_as_pct_of_group_seq_num * number_of_sequences)
#                 uncorrected_min_cluster_size = hdbscan_min_cluster_size
#                 is_corrected = False
#                 if hdbscan_min_cluster_size < 2:
#                     hdbscan_min_cluster_size = 2
#                     is_corrected = True
#                 min_cluster_size_correction_dict[GROUP_FIELD_NAME_STR].append(group)
#                 min_cluster_size_correction_dict[CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_IS_CORRECTED_NAME_STR].append(is_corrected)
#                 min_cluster_size_correction_dict[CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_UNCORRECTED_NAME_STR].append(uncorrected_min_cluster_size)
#                 min_cluster_size_correction_dict[CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE_CORRECTED_NAME_STR].append(hdbscan_min_cluster_size)

#                 clusterer = self.cluster_function(min_cluster_size=hdbscan_min_cluster_size, **self.kwargs)
#             else:
#                 clusterer = self.cluster_function(**self.kwargs)

#             clusterer.fit(square_matrix.astype(float))
#             cluster_labels = clusterer.labels_
#             clustered = (cluster_labels >= 0)
#             percentage_clustered = sum(clustered) / len(cluster_labels) * 100

#             # without unclustered sequences
#             cluster_labels_only_clustered = cluster_labels[clustered] 
#             clusters_only_clustered, cluster_sizes_only_clustered = np.unique(cluster_labels_only_clustered, return_counts=True)
#             number_of_clusters = len(clusters_only_clustered) 
#             cluster_labels = clusterer.labels_.astype(str)
#             clustering_algorithm = clusterer.__class__.__name__

#             # cluster size stats per group
#             mean_cluster_size = np.mean(cluster_sizes_only_clustered)
#             median_cluster_size = np.median(cluster_sizes_only_clustered)
#             min_cluster_size = np.min(cluster_sizes_only_clustered)
#             max_cluster_size = np.max(cluster_sizes_only_clustered)
#             std_cluster_size = np.std(cluster_sizes_only_clustered)
#             iqr_cluster_size = iqr(cluster_sizes_only_clustered)

#             # intermediate results
#             self._user_cluster_mapping_per_group[group] = {user:cluster for user, cluster in zip(users, cluster_labels)}
#             self._user_sequence_id_mapping_per_group[group] = {user:sequence_id for user, sequence_id in zip(users, sequence_ids)}
#             self._user_sequence_array_mapping_per_group[group] = {user:sequence_arr for user, sequence_arr in zip(users, sequence_arrays)}
#             self._user_sequence_length_mapping_per_group[group] = {user:sequence_len for user, sequence_len in zip(users, sequence_lengths)}
#             self._clustered_per_group[group] = clustered
#             self._percentage_clustered_per_group[group] = percentage_clustered
        
#             # create a dataframe containing information about user, associated cluster and evaluation metric for the user per group
#             if self.group_field:
#                 user_cluster_eval_metric_df = self.interactions.copy().loc[self.interactions[self.group_field]==group, :]
#             else:
#                 user_cluster_eval_metric_df = self.interactions.copy()
#             user_cluster_eval_metric_df.insert(1, CLUSTER_FIELD_NAME_STR, user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_cluster_mapping_per_group[group][x]))
        
#             user_cluster_eval_metric_df = (user_cluster_eval_metric_df.groupby([CLUSTER_FIELD_NAME_STR, self.user_field])[self.evaluation_field]
#                                                     .first().reset_index())
#             user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_id_mapping_per_group[group][x])
#             user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_array_mapping_per_group[group][x])
#             user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_length_mapping_per_group[group][x])

#             # remove unclustered user sequences from the instance attribute
#             self._user_cluster_eval_metric_df_per_group[group] = user_cluster_eval_metric_df.loc[user_cluster_eval_metric_df[CLUSTER_FIELD_NAME_STR] != '-1']

#             # data per cluster dict initialization 
#             cluster_list = []
#             number_of_sequences_per_cluster_list = []
#             number_of_unique_sequences_per_cluster_list = []
#             users_per_cluster_list = []
#             sequence_ids_per_cluster_list = []
#             sequence_lengths_per_cluster_list = []
#             sequences_per_cluster_list = []

#             # cluster eval metric stats per group
#             mean_cluster_eval_metric_list = []
#             median_cluster_eval_metric_list = []
#             min_cluster_eval_metric_list = []
#             max_cluster_eval_metric_list = []
#             std_cluster_eval_metric_list = []
#             iqr_cluster_eval_metric_list = []

#             # residuals for normality test
#             residual_list = []

#             # loop over cluster to fill initialized lists
#             for cluster, df in user_cluster_eval_metric_df.groupby(CLUSTER_FIELD_NAME_STR):

#                 cluster_list.append(cluster)
#                 number_of_sequences_per_cluster_list.append(len(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]))
#                 number_of_unique_sequences_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].nunique())
#                 users_per_cluster_list.append(df[self.user_field].values)
#                 sequence_ids_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].values)
#                 sequence_lengths_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR].values)
#                 sequences_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR].values)

#                 # eval metric stats
#                 mean_cluster_eval_metric = df[self.evaluation_field].mean()

#                 mean_cluster_eval_metric_list.append(mean_cluster_eval_metric)
#                 median_cluster_eval_metric_list.append(df[self.evaluation_field].median())
#                 min_cluster_eval_metric_list.append(df[self.evaluation_field].min())
#                 max_cluster_eval_metric_list.append(df[self.evaluation_field].max())
#                 std_cluster_eval_metric_list.append(df[self.evaluation_field].std())
#                 iqr_cluster_eval_metric_list.append(iqr(df[self.evaluation_field]))

#                 # residuals of evaluation metric for normality test -> do not calculate for non clustered sequences
#                 if cluster != '-1':
#                     residuals = list(df[self.evaluation_field] - mean_cluster_eval_metric)
#                     residual_list.extend(residuals)

            
#             # only perform tests if there are at least 2 clusters per group
#             if number_of_clusters >= 2:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("error")
#                     # test for normality of residuals of evaluation metric over all clusters
#                     try:
#                         shapiro_test = pg.normality(residual_list, method='shapiro', alpha=0.05)
#                         shapiro_test_pval = shapiro_test['pval'][0]
#                     except:
#                         shapiro_test_pval = None

#                     try: 
#                         jarque_bera_test = pg.normality(residual_list, method='jarque_bera', alpha=0.05)
#                         jarque_bera_test_pval = jarque_bera_test['pval'][0]
#                     except:
#                         jarque_bera_test_pval = None

#                     try:
#                         agostino_pearson_test = pg.normality(residual_list, method='normaltest', alpha=0.05)
#                         agostino_pearson_test_pval = agostino_pearson_test['pval'][0]
#                     except:
#                         agostino_pearson_test_pval = None

#                     # test for homoscedasticity of evaluation metric between clusters  
#                     try:
#                         bartlett_test = pg.homoscedasticity(self._user_cluster_eval_metric_df_per_group[group], 
#                                                             dv=self.evaluation_field, 
#                                                             group=CLUSTER_FIELD_NAME_STR, 
#                                                             method='bartlett')
#                         bartlett_test_pval = bartlett_test['pval'][0]
#                     except: 
#                         bartlett_test_pval = None

#                     try:
#                         levene_test = pg.homoscedasticity(self._user_cluster_eval_metric_df_per_group[group], 
#                                                           dv=self.evaluation_field, 
#                                                           group=CLUSTER_FIELD_NAME_STR, 
#                                                           method='levene')
#                         levene_test_pval = levene_test['pval'][0]
#                     except:
#                         levene_test_pval = None

#                 # choose omnibus central differences test accordingly to normality and homoscedasticity test results (which test assumptions)
#                 if (shapiro_test_pval > self.normality_test_alpha) and (levene_test_pval > self.homoscedasticity_test_alpha):
#                     central_tendency_test = pg.anova(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR, detailed=True)
#                     central_tendency_test_pval = central_tendency_test['p-unc'][0]
#                     central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_ANOVA_STR
#                 elif (shapiro_test_pval > self.normality_test_alpha) and (levene_test_pval <= self.homoscedasticity_test_alpha):
#                     central_tendency_test = pg.welch_anova(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR)
#                     central_tendency_test_pval = central_tendency_test['p-unc'][0]
#                     central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_WELCH_ANOVA_STR
#                 else: 
#                     central_tendency_test = pg.kruskal(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR, detailed=True)
#                     central_tendency_test_pval = central_tendency_test['p-unc'][0]
#                     central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_KRUSKAL_WALLIS_STR
#             else:
#                 shapiro_test_pval = None
#                 jarque_bera_test_pval = None
#                 agostino_pearson_test_pval = None
#                 bartlett_test_pval = None
#                 levene_test_pval = None
#                 central_tendency_test_pval = None
#                 central_tendency_test_method = None

#             # fill the results dictionaries
#             results_dict_per_group[CLUSTERING_DATASET_NAME_FIELD_NAME_STR].append(self.dataset_name)
#             results_dict_per_group[CLUSTERING_GROUP_FIELD_NAME_STR].append(group)
#             results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR].append(number_of_sequences)
#             results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR].append(number_of_unique_sequences)
#             if self.use_normalized_sequence_distance:
#                 results_dict_per_group[CLUSTERING_MEAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(mean_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(median_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MIN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(min_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MAX_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(max_sequence_metric)
#                 results_dict_per_group[CLUSTERING_STD_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(std_sequence_metric)
#                 results_dict_per_group[CLUSTERING_IQR_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(iqr_sequence_metric)
#             else:
#                 results_dict_per_group[CLUSTERING_MEAN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(mean_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MEDIAN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(median_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MIN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(min_sequence_metric)
#                 results_dict_per_group[CLUSTERING_MAX_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(max_sequence_metric)
#                 results_dict_per_group[CLUSTERING_STD_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(std_sequence_metric)
#                 results_dict_per_group[CLUSTERING_IQR_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(iqr_sequence_metric)
#             results_dict_per_group[CLUSTERING_ALGORITHM_FIELD_NAME_STR].append(clustering_algorithm)
#             if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
#                 results_dict_per_group[CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR].append(hdbscan_min_cluster_size)
#             results_dict_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR].append(number_of_clusters)
#             results_dict_per_group[CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR].append(percentage_clustered)
#             results_dict_per_group[CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR].append(mean_cluster_size)
#             results_dict_per_group[CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR].append(median_cluster_size)
#             results_dict_per_group[CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR].append(min_cluster_size)
#             results_dict_per_group[CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR].append(max_cluster_size)
#             results_dict_per_group[CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR].append(std_cluster_size)
#             results_dict_per_group[CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR].append(iqr_cluster_size)
#             results_dict_per_group[CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR].append(self.evaluation_field)
#             results_dict_per_group[CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR].append(shapiro_test_pval)
#             results_dict_per_group[CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR].append(jarque_bera_test_pval)
#             results_dict_per_group[CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR].append(agostino_pearson_test_pval)
#             results_dict_per_group[CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR].append(levene_test_pval)
#             results_dict_per_group[CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR].append(bartlett_test_pval)
#             results_dict_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR].append(central_tendency_test_method)
#             results_dict_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR].append(central_tendency_test_pval)

#             cluster_stats_results_dict_per_group[CLUSTERING_DATASET_NAME_FIELD_NAME_STR].extend([self.dataset_name] * len(cluster_list))
#             cluster_stats_results_dict_per_group[CLUSTERING_GROUP_FIELD_NAME_STR].extend([group] * len(cluster_list))
#             cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR].extend([number_of_sequences] * len(cluster_list))
#             cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR].extend([number_of_unique_sequences] * len(cluster_list))
#             cluster_stats_results_dict_per_group[CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR].extend([percentage_clustered] * len(cluster_list))
#             cluster_stats_results_dict_per_group[CLUSTER_FIELD_NAME_STR].extend(cluster_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR].extend(number_of_sequences_per_cluster_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR].extend(number_of_unique_sequences_per_cluster_list)
#             cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR].extend(users_per_cluster_list)
#             cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].extend(sequence_ids_per_cluster_list)
#             cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR].extend(sequence_lengths_per_cluster_list)
#             cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR].extend(sequences_per_cluster_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_MEAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(mean_cluster_eval_metric_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_MEDIAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(median_cluster_eval_metric_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_MIN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(min_cluster_eval_metric_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_MAX_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(max_cluster_eval_metric_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_STD_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(std_cluster_eval_metric_list)
#             cluster_stats_results_dict_per_group[CLUSTERING_IQR_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(iqr_cluster_eval_metric_list)

#         # result dataframes -> instance attributes
#         self.cluster_eval_metric_central_tendency_differences_per_group = pd.DataFrame(results_dict_per_group)
#         self.cluster_stats_per_group = pd.DataFrame(cluster_stats_results_dict_per_group)
#         self.group_cluster_analysis_inclusion_status = pd.DataFrame(group_cluster_analysis_inclusion_status_dict)

#         # a dictionary containing information whether the min_cluster_size of hdbscan needed to be corrected to 2
#         if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
#             self.min_cluster_size_correction_df = pd.DataFrame(min_cluster_size_correction_dict) 
#         else:
#             self.min_cluster_size_correction_df = None

#         # flag indicationg whether data is already clustered
#         self._data_clustered = True

#     def perform_post_hoc_pairwise_eval_metric_diff_tests(self,
#                                                          group_str: str,
#                                                          alternative: str):
#         """Performs pairwise post-hoc tests for differences in central tendency of the evaluation metric between clusters. 
#         P-value correction is perfomed via the bonferroni method.

#         Parameters
#         ----------
#         group_str : str
#             A string indicating for which group post-hoc tests will be performed
#         alternative : str
#             A string indicating which type of hypothesis is being tested. Can be either
#                 - two-sided (A != B)
#                 - greater (A > B)
#                 - less (A < B)

#         Returns
#         -------
#         pd.DataFrame
#             A dataframe with post-hoc test results of central tendency differences between each cluster of a group

#         Raises
#         ------
#         Exception
#             If a group has only one cluster of sequence distances, post-hoc tests are not possible
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             data = self._user_cluster_eval_metric_df_per_group[group_str]
#             omnibus_test_type = self.cluster_eval_metric_central_tendency_differences_per_group\
#                                     .loc[self.cluster_eval_metric_central_tendency_differences_per_group\
#                                         [CLUSTERING_GROUP_FIELD_NAME_STR]==group_str, CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR].values[0]

#             number_of_clusters = self.cluster_eval_metric_central_tendency_differences_per_group\
#                                     .loc[self.cluster_eval_metric_central_tendency_differences_per_group\
#                                         [CLUSTERING_GROUP_FIELD_NAME_STR]==group_str, CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR].values[0]
            
#             if number_of_clusters > 1: 
#                 print(f'Post-Hoc Tests for Group: {group_str}')
                
#                 # choose post-hoc tests accordingly to the applied omnibus test for central tendency differences
#                 if omnibus_test_type == CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_KRUSKAL_WALLIS_STR:

#                     print(f'Post-Hoc Pairwise Test: Mann–Whitney U Test')

#                     # mann-whitney U test
#                     post_hoc_results = pg.pairwise_tests(data=data,
#                                                          dv=self.evaluation_field, 
#                                                          between=CLUSTER_FIELD_NAME_STR, 
#                                                          alpha=0.05, 
#                                                          padjust='bonf', 
#                                                          parametric=False,
#                                                          alternative=alternative, 
#                                                          return_desc=True)
#                     post_hoc_results = post_hoc_results.sort_values(by='p-corr')

#                 elif omnibus_test_type == CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_WELCH_ANOVA_STR:

#                     print(f'Post-Hoc Pairwise Test: Welch t-Test')

#                     # welch t-test
#                     post_hoc_results = pg.pairwise_tests(data=data,
#                                                          dv=self.evaluation_field, 
#                                                          between=CLUSTER_FIELD_NAME_STR, 
#                                                          alpha=0.05, 
#                                                          padjust='bonf', 
#                                                          parametric=True,
#                                                          correction=True, 
#                                                          alternative=alternative, 
#                                                          return_desc=True)
#                     post_hoc_results = post_hoc_results.sort_values(by='p-corr')

#                 else:

#                     print(f'Post-Hoc Pairwise Test: t-Test')

#                     # t-test
#                     post_hoc_results = pg.pairwise_tests(data=data,
#                                                          dv=self.evaluation_field, 
#                                                          between=CLUSTER_FIELD_NAME_STR, 
#                                                          alpha=0.05, 
#                                                          padjust='bonf', 
#                                                          parametric=True,
#                                                          correction=False, 
#                                                          alternative=alternative, 
#                                                          return_desc=True)
#                     post_hoc_results = post_hoc_results.sort_values(by='p-corr')
                
#                 return post_hoc_results
            
#             else:
#                 raise Exception(f'Group {group_str} has only one cluster. Post-Hoc Tests are not possible.')
        
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def aggregate_sequence_clustering_and_eval_metric_diff_test_over_groups(self):
#         """Aggregates the clustering results over all groups and inter alia calculates the percentage of groups having\
#         significant differences in central tendency of the evaluation metric between clusters.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             # calculate result fields
#             n_groups_unfiltered = len(self.sequence_distances_dict.keys())
#             n_groups_included = self.cluster_eval_metric_central_tendency_differences_per_group.shape[0]
#             pct_groups_included = n_groups_included / n_groups_unfiltered
#             n_groups_included_multiple_clusters = sum(self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR] > 1)
#             aggregates = self.cluster_eval_metric_central_tendency_differences_per_group\
#                             .groupby([CLUSTERING_DATASET_NAME_FIELD_NAME_STR])\
#                             [[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR, 
#                             CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR, 
#                             CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]]\
#                             .agg([np.mean, np.median, min, max, np.std, iqr])
#             aggregates_multiple_clusters = self.cluster_eval_metric_central_tendency_differences_per_group\
#                                                 .loc[self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR] > 1, :]\
#                                                 .groupby([CLUSTERING_DATASET_NAME_FIELD_NAME_STR])\
#                                                 [[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR, 
#                                                 CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR, 
#                                                 CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]]\
#                                                 .agg([np.mean, np.median, min, max, np.std, iqr])
#             n_groups_included_significant_cluster_central_tendencies_differences = sum(self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR] < 0.05)   
#             percentage_groups_included_significant_cluster_central_tendencies_differences = n_groups_included_significant_cluster_central_tendencies_differences / n_groups_included * 100
#             percentage_groups_included_with_multiple_clusters_significant_cluster_central_tendencies_differences = n_groups_included_significant_cluster_central_tendencies_differences / n_groups_included_multiple_clusters * 100

#             # fill the result dictionary
#             results_dict = {CLUSTERING_UNFILTERED_NUMBER_OF_GROUPS_FIELD_NAME_STR: n_groups_unfiltered,
#                             CLUSTERING_NUMBER_OF_GROUPS_FIELD_NAME_STR: n_groups_included,
#                             CLUSTERING_PCT_OF_UNFILTERED_GROUPS_FIELD_NAME_STR: pct_groups_included,
#                             CLUSTERING_NUMBER_OF_GROUPS_WITH_MULTIPLE_CLUSTERS_FIELD_NAME_STR: n_groups_included_multiple_clusters,
#                             CLUSTERING_MEAN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_MEAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_MEAN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_MEAN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_MEAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_MEAN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['mean'],
#                             CLUSTERING_MEDIAN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['median'],
#                             CLUSTERING_MIN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['min'],
#                             CLUSTERING_MAX_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['max'],
#                             CLUSTERING_STD_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['std'],
#                             CLUSTERING_IQR_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['iqr'],
#                             CLUSTERING_NUMBER_OF_GROUPS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: n_groups_included_significant_cluster_central_tendencies_differences,
#                             CLUSTERING_PCT_OF_GROUPS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: percentage_groups_included_significant_cluster_central_tendencies_differences,
#                             CLUSTERING_PCT_OF_GROUPS_WITH_MULTIPLE_CLUSTERS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: percentage_groups_included_with_multiple_clusters_significant_cluster_central_tendencies_differences}

#             # result dataframe -> instance attribute
#             self.aggregate_sequence_clustering_and_eval_metric_diff_test = pd.DataFrame(results_dict)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_included_groups(self):
#         """Displays the absolute value and percentages of groups included in the cluster analysis.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             n_groups = self.group_cluster_analysis_inclusion_status.shape[0]
#             included_groups = self.group_cluster_analysis_inclusion_status[CLUSTERING_GROUP_INCLUDED_IN_CLUSTER_ANALYSIS_NAME_STR].sum()
#             not_included_groups = n_groups - included_groups
#             pct_included_groups = (included_groups / n_groups) * 100
#             pct_not_included_groups = (not_included_groups / n_groups) * 100

#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTER_FIELD_NAME_STR} Analysis - Included {GROUP_FIELD_NAME_STR}s')
#             print(DASH_STRING)
#             print(' ')
#             print(f'{included_groups} {GROUP_FIELD_NAME_STR}s out of {n_groups} are included in {CLUSTER_FIELD_NAME_STR} Analysis:')
#             g = sns.barplot(x=[f'Included', f'Not Included'], y=[included_groups, not_included_groups])
#             g.set(ylabel=f'Number of {GROUP_FIELD_NAME_STR}s');
#             plt.show()

#             print(f'{pct_included_groups}% of {GROUP_FIELD_NAME_STR}s are included in {CLUSTER_FIELD_NAME_STR} Analysis:')
#             g = sns.barplot(x=[f'Included', f'Not Included'], y=[pct_included_groups, pct_not_included_groups])
#             g.set(ylim=(0, 100), ylabel=f'% of {GROUP_FIELD_NAME_STR}s');
#             plt.show()
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_clusters_by_group_umap(self,
#                                        group_str: str,
#                                        **kwargs):
#         """Reduces the sequence distance matrix of specified group with UMAP and displays clustering in a two-dimensional projection.

#         Parameters
#         ----------
#         group_str : str
#             A string indicating for which group the sequence distance clustering will be displayed
#         **kwargs
#             Keyword arguments for the UMAP class. Random_state and verbose are preset.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             reducer = umap.UMAP(random_state=1,
#                                 verbose = False,
#                                 **kwargs)

#             embedding_2D = reducer.fit_transform(self._square_matrix_per_group[group_str])

#             clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group_str].values()), dtype=int)[self._clustered_per_group[group_str]]

#             g = sns.scatterplot(x = embedding_2D[~self._clustered_per_group[group_str], 0], y = embedding_2D[~self._clustered_per_group[group_str], 1], s=100, marker=".", alpha =1, color="black")
#             g = sns.scatterplot(x = embedding_2D[self._clustered_per_group[group_str], 0], y = embedding_2D[self._clustered_per_group[group_str], 1], s=100, hue=clustered_labels, marker=".", alpha =0.5, palette=sns.husl_palette(len(np.unique(clustered_labels))))
#             g.set(xlabel='Embedding 1',
#                 ylabel='Embedding 2')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title = "Clusters", title_fontsize = 20);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_clusters_by_group_pca(self,
#                                       group_str: str):
#         """Reduces the sequence distance matrix of specified group with PCA and displays clustering in a two-dimensional projection.

#         Parameters
#         ----------
#         group_str : str
#             A string indicating for which group the sequence distance clustering will be displayed

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             reducer = PCA(n_components=2)
#             embedding_2D = reducer.fit_transform(self._square_matrix_per_group[group_str])

#             clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group_str].values()), dtype=int)[self._clustered_per_group[group_str]]

#             g = sns.scatterplot(x = embedding_2D[~self._clustered_per_group[group_str], 0], y = embedding_2D[~self._clustered_per_group[group_str], 1], s=100, marker=".", alpha =1, color="black")
#             g = sns.scatterplot(x = embedding_2D[self._clustered_per_group[group_str], 0], y = embedding_2D[self._clustered_per_group[group_str], 1], s=100, hue=clustered_labels, marker=".", alpha =0.5, palette=sns.husl_palette(len(np.unique(clustered_labels))))
#             g.set(xlabel='Component 1',
#                 ylabel='Component 2')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title = "Clusters", title_fontsize = 20);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_eval_metric_dist_between_clusters_by_group(self,
#                                                            group_str: str):
#         """Displays the evaluation metric distribution per cluster via boxplots, violinplot, boxenplot and kde for specified group.

#         Parameters
#         ----------
#         group_str : str
#             A string indicating for which group the evaluation metric distribution per cluster will be displayed

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             g = sns.boxplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field, 
#                             showmeans=True, 
#                             meanprops=marker_config,
#                             showfliers=False)
#             g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field, 
#                             size=2, 
#                             color="red")
#             plt.show()

#             g = sns.violinplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field,
#                             showmeans=True, 
#                             meanprops=marker_config,
#                             showfliers=False)
#             g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field, 
#                             size=2, 
#                             color="red")
#             plt.show()

#             g = sns.boxenplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR,
#                             y=self.evaluation_field,
#                             showfliers=False)
#             g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field, 
#                             size=2, 
#                             color="red")
#             plt.show()

#             g = sns.displot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
#                             x=self.evaluation_field, 
#                             hue=CLUSTER_FIELD_NAME_STR, 
#                             kind='kde')
#             plt.show()

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')
    

#     def display_cluster_size_by_group(self,
#                                       group_str: str):
#         """Displays the cluster sizes for specified group

#         Parameters
#         ----------
#         group_str : str
#             A string indicating for which group the evaluation metric distribution per cluster will be displayed

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTER_FIELD_NAME_STR} Size for {GROUP_FIELD_NAME_STR} {group_str}:')
#             print(DASH_STRING)
#             g = sns.barplot(data=self.cluster_stats_per_group.loc[self.cluster_stats_per_group[GROUP_FIELD_NAME_STR]==group_str, :], 
#                             x=CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
#                             y=CLUSTER_FIELD_NAME_STR)
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')
    
#     def display_eval_metric_dist_between_cluster_all_groups(self,
#                                                             height: int):
#         """Displays the evaluation metric distribution per cluster via boxplots for all groups.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """

#         if self._data_clustered:        
#             # concat user_cluster_eval_metric_df over all groups
#             user_cluster_eval_metric_df = pd.DataFrame()            
#             for k,v in self._user_cluster_eval_metric_df_per_group.items():
#                 v[GROUP_FIELD_NAME_STR] = k
#                 user_cluster_eval_metric_df = pd.concat([user_cluster_eval_metric_df, v])
            
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'Central Tendency Differences in {CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR} between {CLUSTER_FIELD_NAME_STR}s per {GROUP_FIELD_NAME_STR}:')
#             print(f'Chosen {CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR}: "{self.evaluation_field}"')
#             print(DASH_STRING)
#             g = sns.FacetGrid(user_cluster_eval_metric_df, 
#                               col=GROUP_FIELD_NAME_STR, 
#                               col_wrap=6, 
#                               sharex=False,
#                               sharey=False,
#                               height=height, 
#                               aspect= 1)
#             g.map_dataframe(sns.boxplot, 
#                             x=CLUSTER_FIELD_NAME_STR, 
#                             y=self.evaluation_field,
#                             showmeans=True, 
#                             meanprops=marker_config_eval_metric_mean)
#             g.set(xlabel=CLUSTER_FIELD_NAME_STR, 
#                   ylabel=CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR)
#             for ax in g.axes.flatten():
#                 ax.tick_params(labelbottom=True)
#             g.fig.subplots_adjust(top=0.95)
#             g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_clusters_all_group_umap(self,
#                                         height: int,
#                                         **kwargs):
#         """Reduces the sequence distance matrices of all group with UMAP and displays clustering in a two-dimensional projection.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots
#         **kwargs
#             Keyword arguments for the UMAP class. Random_state and verbose are preset.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             reducer = umap.UMAP(random_state=1,
#                                 verbose = False,
#                                 **kwargs)

#             emb_per_group_df = pd.DataFrame()

#             n_clusters_per_group = []
#             for group, distance_matrix in self._square_matrix_per_group.items():

#                 embedding_2D = reducer.fit_transform(distance_matrix)

#                 clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group].values()), dtype=int)[self._clustered_per_group[group]]
#                 n_clusters_per_group.append(len(np.unique(clustered_labels)))

#                 emb_clustered = embedding_2D[self._clustered_per_group[group], :]
#                 emb_not_clustered = embedding_2D[~self._clustered_per_group[group], :]
                
#                 emb_clustered_df = pd.DataFrame(emb_clustered, columns=['x_clust', 'y_clust'])
#                 emb_clustered_df[CLUSTER_FIELD_NAME_STR] = clustered_labels
#                 emb_not_clustered_df = pd.DataFrame(emb_not_clustered, columns=['x_not_clust', 'y_not_clust'])
#                 emb_df = pd.concat([emb_clustered_df, emb_not_clustered_df], ignore_index=False, axis=1)
#                 emb_df[GROUP_FIELD_NAME_STR] = group

#                 emb_per_group_df = pd.concat([emb_per_group_df, emb_df])

#             max_n_clusters = np.max(n_clusters_per_group)

#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTER_FIELD_NAME_STR}s per {GROUP_FIELD_NAME_STR}')
#             print(f'Dimensionality Reducer: UMAP')
#             print(DASH_STRING)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 g = sns.FacetGrid(emb_per_group_df, 
#                                 col=GROUP_FIELD_NAME_STR, 
#                                 col_wrap=6, 
#                                 sharex=False,
#                                 sharey=False,
#                                 height=height, 
#                                 aspect= 1)
#                 g.map_dataframe(sns.scatterplot, 
#                                 x='x_not_clust', 
#                                 y='y_not_clust',
#                                 color='black',
#                                 alpha=1,
#                                 s=10)
#                 g.map_dataframe(sns.scatterplot, 
#                                 x='x_clust', 
#                                 y='y_clust',
#                                 hue=CLUSTER_FIELD_NAME_STR,
#                                 palette=sns.color_palette("hls", max_n_clusters),
#                                 alpha=1,
#                                 s=10)
#                 g.set(xlabel='Embedding 1', 
#                     ylabel='Embedding 2')
#                 for ax in g.axes.flatten():
#                     ax.tick_params(labelbottom=True)
#                 g.fig.subplots_adjust(top=0.95)
#                 g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#                 plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_clusters_all_group_pca(self,
#                                        height: int):
#         """Reduces the sequence distance matrices of all group with PCA and displays clustering in a two-dimensional projection.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             reducer = PCA(n_components=2)

#             emb_per_group_df = pd.DataFrame()

#             n_clusters_per_group = []
#             for group, distance_matrix in self._square_matrix_per_group.items():

#                 embedding_2D = reducer.fit_transform(distance_matrix)

#                 clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group].values()), dtype=int)[self._clustered_per_group[group]]
#                 n_clusters_per_group.append(len(np.unique(clustered_labels)))

#                 emb_clustered = embedding_2D[self._clustered_per_group[group], :]
#                 emb_not_clustered = embedding_2D[~self._clustered_per_group[group], :]
                
#                 emb_clustered_df = pd.DataFrame(emb_clustered, columns=['x_clust', 'y_clust'])
#                 emb_clustered_df[CLUSTER_FIELD_NAME_STR] = clustered_labels
#                 emb_not_clustered_df = pd.DataFrame(emb_not_clustered, columns=['x_not_clust', 'y_not_clust'])
#                 emb_df = pd.concat([emb_clustered_df, emb_not_clustered_df], ignore_index=False, axis=1)
#                 emb_df[GROUP_FIELD_NAME_STR] = group

#                 emb_per_group_df = pd.concat([emb_per_group_df, emb_df])

#             max_n_clusters = np.max(n_clusters_per_group)

#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTER_FIELD_NAME_STR}s per {GROUP_FIELD_NAME_STR}')
#             print(f'Dimensionality Reducer: PCA')
#             print(DASH_STRING)
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 g = sns.FacetGrid(emb_per_group_df, 
#                                 col=GROUP_FIELD_NAME_STR, 
#                                 col_wrap=6, 
#                                 sharex=False,
#                                 height=height, aspect= 1)
#                 g.map_dataframe(sns.scatterplot, 
#                                 x='x_not_clust', 
#                                 y='y_not_clust',
#                                 color='black',
#                                 alpha=1,
#                                 s=10)
#                 g.map_dataframe(sns.scatterplot, 
#                                 x='x_clust', 
#                                 y='y_clust',
#                                 hue=CLUSTER_FIELD_NAME_STR,
#                                 palette=sns.color_palette("hls", max_n_clusters),
#                                 alpha=1,
#                                 s=10)
#                 g.set(xlabel='Embedding 1', 
#                     ylabel='Embedding 2')
#                 for ax in g.axes.flatten():
#                     ax.tick_params(labelbottom=True)
#                 g.fig.subplots_adjust(top=0.95)
#                 g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#                 plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_number_of_clusters_all_group(self):
#         """Displays the number of clusters per group as barplot.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
#                             x=CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR,
#                             y=GROUP_FIELD_NAME_STR)
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_min_cluster_size_all_group(self):
#         """Displays the size of the smallest cluster per group as barplot.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
#                             x=CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR,
#                             y=GROUP_FIELD_NAME_STR)
#             plt.show(g)
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_percentage_clustered_all_group(self):
#         """Displays the percentage of sequencese clustered per group as barplot.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
#                             x=CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR,
#                             y=GROUP_FIELD_NAME_STR)
#             plt.show(g)
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_number_sequences_per_cluster_all_group(self,
#                                                        height: int):
#         """Displays the number of sequences per cluster for each group as multi-grid barplot.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'Number of {SEQUENCE_STR}s per {CLUSTER_FIELD_NAME_STR} for each {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(f'Plots:')
#             g = sns.FacetGrid(self.cluster_stats_per_group, 
#                               col=GROUP_FIELD_NAME_STR, 
#                               col_wrap=6, 
#                               sharex=False,
#                               sharey=False,
#                               height=height, 
#                               aspect= 1)
#             g.map_dataframe(sns.barplot, 
#                             x=CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
#                             y=CLUSTER_FIELD_NAME_STR)
#             for ax in g.axes.flatten():
#                 ax.tick_params(labelbottom=True)
#             g.fig.subplots_adjust(top=0.95)
#             g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_number_unique_sequences_per_cluster_all_group(self,
#                                                               height: int):
#         """Displays the number of unique sequences per cluster for each group as multi-grid barplot.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'Number of Unique {SEQUENCE_STR}s per {CLUSTER_FIELD_NAME_STR} for each {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(f'Plots:')
#             g = sns.FacetGrid(self.cluster_stats_per_group, 
#                               col=GROUP_FIELD_NAME_STR, 
#                               col_wrap=6, 
#                               sharex=False,
#                               sharey=False,
#                               height=height, 
#                               aspect= 1)
#             g.map_dataframe(sns.barplot, 
#                             x=CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
#                             y=CLUSTER_FIELD_NAME_STR)
#             index = CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR.find(SEQUENCE_STR)
#             xlabel = CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR[:index] + '\n' + CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR[index:]
#             g.set_axis_labels(x_var=xlabel)
#             for ax in g.axes.flatten():
#                 ax.tick_params(labelbottom=True)
#             g.fig.subplots_adjust(top=0.95)
#             g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def display_number_unique_sequences_vs_number_sequences_per_cluster_all_group(self,
#                                                                                   height: int):
#         """Displays the relationship between the number of unique sequences vs the number of sequences per cluster for each group as multi-grid scatterplot.

#         Parameters
#         ----------
#         height : int
#             The height of the subplots

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'Number of Unique {SEQUENCE_STR}s vs Number of {SEQUENCE_STR}s per {CLUSTER_FIELD_NAME_STR} for each {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(f'Plots:')
#             g = sns.FacetGrid(self.cluster_stats_per_group, 
#                               col=GROUP_FIELD_NAME_STR, 
#                               col_wrap=6, 
#                               sharex=False,
#                               sharey=False,
#                               height=height, 
#                               aspect= 1)
#             g.map_dataframe(sns.scatterplot, 
#                             x=CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
#                             y=CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR)
#             index = CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR.find(SEQUENCE_STR)
#             ylabel = CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR[:index] + '\n' + CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR[index:]
#             g.set_axis_labels(y_var=ylabel)
#             for ax in g.axes.flatten():
#                 ax.tick_params(labelbottom=True)
#             g.fig.subplots_adjust(top=0.95)
#             g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
#             plt.show(g)

#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def print_number_of_clusters_all_group(self):
#         """Print the number of clusters for each group.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR} for each {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(self.cluster_eval_metric_central_tendency_differences_per_group[[GROUP_FIELD_NAME_STR, CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]]\
#                   .sort_values(by=CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR)\
#                   .to_string(index=False))
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def print_min_cluster_sizes_all_group(self):
#         """Print the minimum cluster sizes sizes for each group.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(self.cluster_eval_metric_central_tendency_differences_per_group[[GROUP_FIELD_NAME_STR, CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR]]\
#                   .sort_values(by=CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR)\
#                   .to_string(index=False))
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

#     def print_percentage_clustered_all_group(self):
#         """Print the percentage of sequences clustered for each group.

#         Raises
#         ------
#         Exception
#             Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
#         """
#         if self._data_clustered:        
#             print(STAR_STRING)
#             print(STAR_STRING)
#             print(' ')
#             print(DASH_STRING)
#             print(f'{CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
#             print(DASH_STRING)
#             print(self.cluster_eval_metric_central_tendency_differences_per_group[[GROUP_FIELD_NAME_STR, CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR]]\
#                   .sort_values(by=CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR)\
#                   .to_string(index=False))
#         else:
#             raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')
