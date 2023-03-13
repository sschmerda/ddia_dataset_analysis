from .standard_import import *
from .constants import *
from .config import *

class ClusterEvaluation:
    """
    A class containing methods which perform several tasks:
        1. Calculation of sequence distance clusters for each group

        2. Performance of omnibus testing for central tendency differences in a specified evaluation metric
           between each cluster
            - Depending on normality and homoscedasticity of the evaluation metric in the clusters an appropriate test
              will be chosen: Anova vs Welch-Anova vs Kruskall-Wallis
        
        3. Performance of post-hoc tests for differences between single clusters of a group

        4. Calculation of sequence and cluster statistics for each group

        5. Plotting of cluster results and differneces in distribution of specified evaluation metric between clusters

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.
    interactions: pd.DataFrame
        The interactions dataframe.
    user_field: str
        Then name of the user field.
    group_field: str
        Then name of the group field.
        This argument should be set to None if the interactions dataframe does not have a group_field.
    evaluation_field: str
        Then name of the evaluation metric field for which central tendency differences between clusters are being tested.
    sequence_distances_dict: dict
        A nested dictionary containing results of "get_user_sequence_distances_per_group" or "get_user_sequence_distances" methods\
        of SeqDist or SeqDistNoGroup for all available groups.
        For each group the subdictionary must contain the following keys: 
        ('Sequence Distance', 'Sequence Maximum Length', 'Sequence User Combination', 'User', 'Sequence Length', 'Sequence ID', 'Sequence Array').
    use_normalized_sequence_distance: bool
        A flag indicating whether a normalized sequence distance, ranging from 0 to 1, will be used for clustering.
    cluster_function
        A scikit-learn compatible (syntax- and method-wise) cluster algorithm.
    hdbscan_min_cluster_size_as_pct_of_group_seq_num: float, None
        The minimal cluster size as percentage of number of sequences per group. This parameters only applies if\
        cluster_function is equal to the HDBSCAN class of the hdbscan library and otherwise should be set to
        None.
    normality_test_alpha: float
        A significance level for the normality test (shapiro) -> used for testing anova assumptions.
    homoscedasticity_test_alpha: float
        A significance level for the homoscedasticity test (levene) -> used for testing anova assumptions.
    **kwargs
        Keyword arguments functioning as parameters for the specified cluster_function.

    Methods
    -------
    cluster_sequences_and_test_eval_metric_diff   
        Clusters the user sequence distances for each group and performs omnibus test for central tendecy differences between the clusters.

    perform_post_hoc_pairwise_eval_metric_diff_tests(group_str: str
                                                     alternative: str)
        Performs pairwise post-hoc tests for differences in central tendency of the evaluation metric between clusters.
        
    aggregate_sequence_clustering_and_eval_metric_diff_test_over_groups
        Aggregates the clustering results over all groups and inter alia calculates the percentage of groups having\
        significant differences in central tendency of the evaluation metric between clusters.

    display_clusters_by_group_umap(group_str: str
                                   **kwargs)
        Reduces the sequence distance matrix of specified group with UMAP and displays clustering in a two-dimensional projection.

    display_clusters_by_group_pca(group_str: str)
        Reduces the sequence distance matrix of specified group with PCA and displays clustering in a two-dimensional projection.
    
    display_eval_metric_dist_between_clusters_by_group(group_str: str)
        Displays the evaluation metric distribution per cluster via boxplots, violinplot, boxenplot and kde for specified group.
    
    display_cluster_size_by_group(group_str: str)
        Displays the cluster sizes for specified group
    
    display_eval_metric_dist_between_cluster_all_groups
        Displays the evaluation metric distribution per cluster via boxplots for all groups.

    display_clusters_all_group_umap
        Reduces the sequence distance matrices of all group with UMAP and displays clustering in a two-dimensional projection.

    display_clusters_all_group_pca
        Reduces the sequence distance matrices of all group with PCA and displays clustering in a two-dimensional projection.

    display_number_of_clusters_all_group
        Displays the number of clusters per group as barplot.

    display_min_cluster_size_all_group:
        Displays the size of the smallest cluster per group as barplot.
    
    display_percentage_clustered_all_group
        Displays the percentage of sequencese clustered per group as barplot.

    display_cluster_size_all_group
        Displays the cluster sizes for each group as multi-grid barplot.

    print_min_cluster_sizes_all_group
        Print the minimum cluster sizes sizes for each group.

    print_percentage_clustered_all_group
        Print the percentage of sequences clustered for each group.

    Attributes
    -------
    cluster_eval_metric_central_tendency_differences_per_group
        A dataframe containing groupwise sequence and cluster stats, assumptions(normality and homoscedasticity) test results for the omnibus test and
        and omnibus central tendency differences of evaluation metrics between clusters test results

    cluster_stats_per_group
        A dataframe containing cluster information for each group

    aggregate_sequence_clustering_and_eval_metric_diff_test
        A dataframe containing aggregates of cluster_eval_metric_central_tendency_differences_per_group. Contains the main result
        field (percentage of groups with significant central tendency differences in evaluation metric) and includes number of sequence,
        number of unique sequence and number of cluster stats per group.
    """
    def __init__(self, 
                 dataset_name: str,
                 interactions: pd.DataFrame,
                 user_field: str,
                 group_field: str,
                 evaluation_field: str,
                 sequence_distances_dict: dict,
                 use_normalized_sequence_distance: bool,
                 cluster_function,
                 hdbscan_min_cluster_size_as_pct_of_group_seq_num: float,
                 normality_test_alpha: float,
                 homoscedasticity_test_alpha: float,
                 **kwargs):

        self.dataset_name = dataset_name
        self.interactions = interactions
        self.user_field = user_field
        self.group_field = group_field
        self.evaluation_field = evaluation_field
        self.sequence_distances_dict= sequence_distances_dict
        self.use_normalized_sequence_distance = use_normalized_sequence_distance
        self.cluster_function = cluster_function
        self.hdbscan_min_cluster_size_as_pct_of_group_seq_num = hdbscan_min_cluster_size_as_pct_of_group_seq_num
        self.normality_test_alpha = normality_test_alpha 
        self.homoscedasticity_test_alpha = homoscedasticity_test_alpha
        self.kwargs = kwargs

        # flag indicationg whether data is already clustered
        self._data_clustered = False

        # the central results of ClusterEvaluation
        self.cluster_eval_metric_central_tendency_differences_per_group = None
        self.cluster_stats_per_group = None
        self.aggregate_sequence_clustering_and_eval_metric_diff_test = None
        
        # intermediate results
        self._square_matrix_per_group = {}
        self._user_cluster_mapping_per_group = {}
        self._user_sequence_id_mapping_per_group = {}
        self._user_sequence_array_mapping_per_group = {}
        self._user_sequence_length_mapping_per_group = {}
        self._clustered_per_group = {}
        self._percentage_clustered_per_group = {}
        self._user_cluster_eval_metric_df_per_group = {}

    def cluster_sequences_and_test_eval_metric_diff(self):
        """Clusters the user sequence distances for each group and performs omnibus test for central tendecy differences between the clusters.
        """        
        # initialization of result dictionaries used for generating result dataframes
        if self.use_normalized_sequence_distance:
            results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
                                      CLUSTERING_GROUP_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
                                      CLUSTERING_MEAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MIN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MAX_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_STD_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_IQR_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_ALGORITHM_FIELD_NAME_STR:[],
                                      CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR:[],
                                      CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
                                      CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR:[],
                                      CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR:[]}
        else:
            results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
                                      CLUSTERING_GROUP_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
                                      CLUSTERING_MEAN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MEDIAN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MIN_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_MAX_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_STD_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_IQR_SEQUENCE_DISTANCE_FIELD_NAME_STR:[],
                                      CLUSTERING_ALGORITHM_FIELD_NAME_STR:[],
                                      CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR:[],
                                      CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
                                      CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR:[],
                                      CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR:[],
                                      CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR:[],
                                      CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR:[]}
        

        cluster_stats_results_dict_per_group = {CLUSTERING_DATASET_NAME_FIELD_NAME_STR:[],
                                                CLUSTERING_GROUP_FIELD_NAME_STR:[],
                                                CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR:[],
                                                CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR:[],
                                                CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR:[],
                                                CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR:[],
                                                LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR:[],
                                                LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR:[],
                                                LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR:[],
                                                LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR:[],
                                                CLUSTERING_MEAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_MEDIAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_MIN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_MAX_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_STD_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[],
                                                CLUSTERING_IQR_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR:[]}
        
        # if a different cluster algorithm to HDBSCAN is used, delete key for chosen minimal cluster size per group
        if not self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
            _ = results_dict_per_group.pop(CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR)

        # loop over all groups in the sequence distance dictionary
        for group, subdict in tqdm(self.sequence_distances_dict.items()):

            # extract data from dictionary
            users = subdict[LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR]
            distances = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_DISTANCE_NAME_STR])
            max_sequence_len_per_distance = np.array(subdict[LEARNING_ACTIVITY_SEQUENCE_MAX_LENGTH_NAME_STR])
            sequence_ids = subdict[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]
            sequence_arrays = subdict[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR]
            sequence_lengths = subdict[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR]
            number_of_sequences = len(sequence_ids)
            number_of_unique_sequences = len(np.unique(sequence_ids))

            # choose between normalized and non-normalized sequence distance
            if self.use_normalized_sequence_distance:
                distance_array = distances / max_sequence_len_per_distance 
            else:
                distance_array = distances
            
            # sequence distance stats per group
            mean_sequence_metric = np.mean(distance_array)
            median_sequence_metric = np.median(distance_array)
            min_sequence_metric = np.min(distance_array)
            max_sequence_metric = np.max(distance_array)
            std_sequence_metric = np.std(distance_array)
            iqr_sequence_metric = iqr(distance_array)

            # generate distance matrix used for clustering:
            square_matrix = squareform(distance_array)
            self._square_matrix_per_group[group] = square_matrix

            # initialize clusterer object, cluster data and calculate cluster labels for each user sequence
            if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
                hdbscan_min_cluster_size = round(self.hdbscan_min_cluster_size_as_pct_of_group_seq_num * number_of_sequences)
                clusterer = self.cluster_function(min_cluster_size=hdbscan_min_cluster_size, **self.kwargs)
            else:
                clusterer = self.cluster_function(**self.kwargs)

            clusterer.fit(square_matrix.astype(float))
            cluster_labels = clusterer.labels_
            clustered = (cluster_labels >= 0)
            percentage_clustered = sum(clustered) / len(cluster_labels) * 100

            # without unclustered sequences
            cluster_labels_only_clustered = cluster_labels[clustered] 
            clusters_only_clustered, cluster_sizes_only_clustered = np.unique(cluster_labels_only_clustered, return_counts=True)
            number_of_clusters = len(clusters_only_clustered) 
            cluster_labels = clusterer.labels_.astype(str)
            clustering_algorithm = clusterer.__class__.__name__

            # cluster size stats per group
            mean_cluster_size = np.mean(cluster_sizes_only_clustered)
            median_cluster_size = np.median(cluster_sizes_only_clustered)
            min_cluster_size = np.min(cluster_sizes_only_clustered)
            max_cluster_size = np.max(cluster_sizes_only_clustered)
            std_cluster_size = np.std(cluster_sizes_only_clustered)
            iqr_cluster_size = iqr(cluster_sizes_only_clustered)

            # intermediate results
            self._user_cluster_mapping_per_group[group] = {user:cluster for user, cluster in zip(users, cluster_labels)}
            self._user_sequence_id_mapping_per_group[group] = {user:sequence_id for user, sequence_id in zip(users, sequence_ids)}
            self._user_sequence_array_mapping_per_group[group] = {user:sequence_arr for user, sequence_arr in zip(users, sequence_arrays)}
            self._user_sequence_length_mapping_per_group[group] = {user:sequence_len for user, sequence_len in zip(users, sequence_lengths)}
            self._clustered_per_group[group] = clustered
            self._percentage_clustered_per_group[group] = percentage_clustered
            
            # create a dataframe containing information about user, associated cluster and evaluation metric for the user per group
            if self.group_field:
                user_cluster_eval_metric_df = self.interactions.copy().loc[self.interactions[self.group_field]==group, :]
            else:
                user_cluster_eval_metric_df = self.interactions.copy()
            user_cluster_eval_metric_df.insert(1, CLUSTER_FIELD_NAME_STR, user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_cluster_mapping_per_group[group][x]))
            
            user_cluster_eval_metric_df = (user_cluster_eval_metric_df.groupby([CLUSTER_FIELD_NAME_STR, self.user_field])[self.evaluation_field]
                                                    .first().reset_index())
            user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_id_mapping_per_group[group][x])
            user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_array_mapping_per_group[group][x])
            user_cluster_eval_metric_df[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR] = user_cluster_eval_metric_df[self.user_field].apply(lambda x: self._user_sequence_length_mapping_per_group[group][x])

            # remove unclustered user sequences from the instance attribute
            self._user_cluster_eval_metric_df_per_group[group] = user_cluster_eval_metric_df.loc[user_cluster_eval_metric_df[CLUSTER_FIELD_NAME_STR] != '-1']

            # data per cluster dict initialization 
            cluster_list = []
            number_of_sequences_per_cluster_list = []
            number_of_unique_sequences_per_cluster_list = []
            users_per_cluster_list = []
            sequence_ids_per_cluster_list = []
            sequence_lengths_per_cluster_list = []
            sequences_per_cluster_list = []

            # cluster eval metric stats per group
            mean_cluster_eval_metric_list = []
            median_cluster_eval_metric_list = []
            min_cluster_eval_metric_list = []
            max_cluster_eval_metric_list = []
            std_cluster_eval_metric_list = []
            iqr_cluster_eval_metric_list = []

            # residuals for normality test
            residual_list = []

            # loop over cluster to fill initialized lists
            for cluster, df in user_cluster_eval_metric_df.groupby(CLUSTER_FIELD_NAME_STR):

                cluster_list.append(cluster)
                number_of_sequences_per_cluster_list.append(len(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR]))
                number_of_unique_sequences_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].nunique())
                users_per_cluster_list.append(df[self.user_field].values)
                sequence_ids_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].values)
                sequence_lengths_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR].values)
                sequences_per_cluster_list.append(df[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR].values)

                # eval metric stats
                mean_cluster_eval_metric = df[self.evaluation_field].mean()

                mean_cluster_eval_metric_list.append(mean_cluster_eval_metric)
                median_cluster_eval_metric_list.append(df[self.evaluation_field].median())
                min_cluster_eval_metric_list.append(df[self.evaluation_field].min())
                max_cluster_eval_metric_list.append(df[self.evaluation_field].max())
                std_cluster_eval_metric_list.append(df[self.evaluation_field].std())
                iqr_cluster_eval_metric_list.append(iqr(df[self.evaluation_field]))

                # residuals of evaluation metric for normality test -> do not calculate for non clustered sequences
                if cluster != '-1':
                    residuals = list(df[self.evaluation_field] - mean_cluster_eval_metric)
                    residual_list.extend(residuals)

            
            # only perform tests if there are at least 2 clusters per group
            if number_of_clusters >= 2:

                # test for normality of residuals of evaluatiom metric over all clusters
                shapiro_test = pg.normality(residual_list, method='shapiro', alpha=0.05)
                shapiro_test_pval = shapiro_test['pval'][0]

                jarque_bera_test = pg.normality(residual_list, method='jarque_bera', alpha=0.05)
                jarque_bera_test_pval = jarque_bera_test['pval'][0]

                agostino_pearson_test = pg.normality(residual_list, method='normaltest', alpha=0.05)
                agostino_pearson_test_pval = agostino_pearson_test['pval'][0]

                # test for homoscedasticity of evaluation metric between clusters  
                bartlett_test = pg.homoscedasticity(self._user_cluster_eval_metric_df_per_group[group], 
                                                    dv=self.evaluation_field, 
                                                    group=CLUSTER_FIELD_NAME_STR, 
                                                    method='bartlett')
                bartlett_test_pval = bartlett_test['pval'][0]

                levene_test = pg.homoscedasticity(self._user_cluster_eval_metric_df_per_group[group], 
                                                  dv=self.evaluation_field, 
                                                  group=CLUSTER_FIELD_NAME_STR, 
                                                  method='levene')
                levene_test_pval = levene_test['pval'][0]

                # choose omnibus central differences test accordingly to normality and homoscedasticity test results (which test assumptions)
                if (shapiro_test_pval > self.normality_test_alpha) and (levene_test_pval > self.homoscedasticity_test_alpha):
                    central_tendency_test = pg.anova(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR, detailed=True)
                    central_tendency_test_pval = central_tendency_test['p-unc'][0]
                    central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_ANOVA_STR
                elif (shapiro_test_pval > self.normality_test_alpha) and (levene_test_pval <= self.homoscedasticity_test_alpha):
                    central_tendency_test = pg.welch_anova(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR)
                    central_tendency_test_pval = central_tendency_test['p-unc'][0]
                    central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_WELCH_ANOVA_STR
                else: 
                    central_tendency_test = pg.kruskal(data=self._user_cluster_eval_metric_df_per_group[group], dv=self.evaluation_field, between=CLUSTER_FIELD_NAME_STR, detailed=True)
                    central_tendency_test_pval = central_tendency_test['p-unc'][0]
                    central_tendency_test_method = CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_KRUSKAL_WALLIS_STR
            else:
                shapiro_test_pval = None
                jarque_bera_test_pval = None
                agostino_pearson_test_pval = None
                bartlett_test_pval = None
                levene_test_pval = None
                central_tendency_test_pval = None
                central_tendency_test_method = None


            # fill the results dictionaries
            results_dict_per_group[CLUSTERING_DATASET_NAME_FIELD_NAME_STR].append(self.dataset_name)
            results_dict_per_group[CLUSTERING_GROUP_FIELD_NAME_STR].append(group)
            results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR].append(number_of_sequences)
            results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR].append(number_of_unique_sequences)
            if self.use_normalized_sequence_distance:
                results_dict_per_group[CLUSTERING_MEAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(mean_sequence_metric)
                results_dict_per_group[CLUSTERING_MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(median_sequence_metric)
                results_dict_per_group[CLUSTERING_MIN_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(min_sequence_metric)
                results_dict_per_group[CLUSTERING_MAX_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(max_sequence_metric)
                results_dict_per_group[CLUSTERING_STD_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(std_sequence_metric)
                results_dict_per_group[CLUSTERING_IQR_NORMALIZED_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(iqr_sequence_metric)
            else:
                results_dict_per_group[CLUSTERING_MEAN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(mean_sequence_metric)
                results_dict_per_group[CLUSTERING_MEDIAN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(median_sequence_metric)
                results_dict_per_group[CLUSTERING_MIN_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(min_sequence_metric)
                results_dict_per_group[CLUSTERING_MAX_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(max_sequence_metric)
                results_dict_per_group[CLUSTERING_STD_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(std_sequence_metric)
                results_dict_per_group[CLUSTERING_IQR_SEQUENCE_DISTANCE_FIELD_NAME_STR].append(iqr_sequence_metric)
            results_dict_per_group[CLUSTERING_ALGORITHM_FIELD_NAME_STR].append(clustering_algorithm)
            if self.hdbscan_min_cluster_size_as_pct_of_group_seq_num:
                results_dict_per_group[CLUSTERING_HDBSCAN_MIN_CLUST_SIZE_FIELD_NAME_STR].append(hdbscan_min_cluster_size)
            results_dict_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR].append(number_of_clusters)
            results_dict_per_group[CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR].append(percentage_clustered)
            results_dict_per_group[CLUSTERING_MEAN_CLUSTER_SIZE_FIELD_NAME_STR].append(mean_cluster_size)
            results_dict_per_group[CLUSTERING_MEDIAN_CLUSTER_SIZE_FIELD_NAME_STR].append(median_cluster_size)
            results_dict_per_group[CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR].append(min_cluster_size)
            results_dict_per_group[CLUSTERING_MAX_CLUSTER_SIZE_FIELD_NAME_STR].append(max_cluster_size)
            results_dict_per_group[CLUSTERING_STD_CLUSTER_SIZE_FIELD_NAME_STR].append(std_cluster_size)
            results_dict_per_group[CLUSTERING_IQR_CLUSTER_SIZE_FIELD_NAME_STR].append(iqr_cluster_size)
            results_dict_per_group[CLUSTERING_COMPARISON_EVALUATION_METRIC_FIELD_NAME_STR].append(self.evaluation_field)
            results_dict_per_group[CLUSTERING_NORMALITY_TEST_SHAPIRO_PVAL_FIELD_NAME_STR].append(shapiro_test_pval)
            results_dict_per_group[CLUSTERING_NORMALITY_TEST_JARQUE_BERA_PVAL_FIELD_NAME_STR].append(jarque_bera_test_pval)
            results_dict_per_group[CLUSTERING_NORMALITY_TEST_AGOSTINO_PEARSON_PVAL_FIELD_NAME_STR].append(agostino_pearson_test_pval)
            results_dict_per_group[CLUSTERING_HOMOSCEDASTICITY_TEST_LEVENE_PVAL_FIELD_NAME_STR].append(levene_test_pval)
            results_dict_per_group[CLUSTERING_HOMOSCEDASTICITY_TEST_BARTLETT_PVAL_FIELD_NAME_STR].append(bartlett_test_pval)
            results_dict_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR].append(central_tendency_test_method)
            results_dict_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR].append(central_tendency_test_pval)

            cluster_stats_results_dict_per_group[CLUSTERING_DATASET_NAME_FIELD_NAME_STR].extend([self.dataset_name] * len(cluster_list))
            cluster_stats_results_dict_per_group[CLUSTERING_GROUP_FIELD_NAME_STR].extend([group] * len(cluster_list))
            cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR].extend([number_of_sequences] * len(cluster_list))
            cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR].extend([number_of_unique_sequences] * len(cluster_list))
            cluster_stats_results_dict_per_group[CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR].extend([percentage_clustered] * len(cluster_list))
            cluster_stats_results_dict_per_group[CLUSTER_FIELD_NAME_STR].extend(cluster_list)
            cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR].extend(number_of_sequences_per_cluster_list)
            cluster_stats_results_dict_per_group[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR].extend(number_of_unique_sequences_per_cluster_list)
            cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_USER_NAME_STR].extend(users_per_cluster_list)
            cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_ID_NAME_STR].extend(sequence_ids_per_cluster_list)
            cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_LENGTH_NAME_STR].extend(sequence_lengths_per_cluster_list)
            cluster_stats_results_dict_per_group[LEARNING_ACTIVITY_SEQUENCE_ARRAY_NAME_STR].extend(sequences_per_cluster_list)
            cluster_stats_results_dict_per_group[CLUSTERING_MEAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(mean_cluster_eval_metric_list)
            cluster_stats_results_dict_per_group[CLUSTERING_MEDIAN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(median_cluster_eval_metric_list)
            cluster_stats_results_dict_per_group[CLUSTERING_MIN_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(min_cluster_eval_metric_list)
            cluster_stats_results_dict_per_group[CLUSTERING_MAX_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(max_cluster_eval_metric_list)
            cluster_stats_results_dict_per_group[CLUSTERING_STD_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(std_cluster_eval_metric_list)
            cluster_stats_results_dict_per_group[CLUSTERING_IQR_EVAL_METRIC_PER_CLUSTER_FIELD_NAME_STR].extend(iqr_cluster_eval_metric_list)

        # result dataframes -> instance attributes
        self.cluster_eval_metric_central_tendency_differences_per_group = pd.DataFrame(results_dict_per_group)
        self.cluster_stats_per_group = pd.DataFrame(cluster_stats_results_dict_per_group)

        # flag indicationg whether data is already clustered
        self._data_clustered = True

    def perform_post_hoc_pairwise_eval_metric_diff_tests(self,
                                                         group_str: str,
                                                         alternative: str):
        """Performs pairwise post-hoc tests for differences in central tendency of the evaluation metric between clusters. 
        P-value correction is perfomed via the bonferroni method.

        Parameters
        ----------
        group_str : str
            A string indicating for which group post-hoc tests will be performed
        alternative : str
            A string indicating which type of hypothesis is being tested. Can be either
                - two-sided (A != B)
                - greater (A > B)
                - less (A < B)

        Returns
        -------
        pd.DataFrame
            A dataframe with post-hoc test results of central tendency differences between each cluster of a group

        Raises
        ------
        Exception
            If a group has only one cluster of sequence distances, post-hoc tests are not possible
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            data = self._user_cluster_eval_metric_df_per_group[group_str]
            omnibus_test_type = self.cluster_eval_metric_central_tendency_differences_per_group\
                                    .loc[self.cluster_eval_metric_central_tendency_differences_per_group\
                                        [CLUSTERING_GROUP_FIELD_NAME_STR]==group_str, CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_TYPE_FIELD_NAME_STR].values[0]

            number_of_clusters = self.cluster_eval_metric_central_tendency_differences_per_group\
                                    .loc[self.cluster_eval_metric_central_tendency_differences_per_group\
                                        [CLUSTERING_GROUP_FIELD_NAME_STR]==group_str, CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR].values[0]
            
            if number_of_clusters > 1: 
                print(f'Post-Hoc Tests for Group: {group_str}')
                
                # choose post-hoc tests accordingly to the applied omnibus test for central tendency differences
                if omnibus_test_type == CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_KRUSKAL_WALLIS_STR:

                    print(f'Post-Hoc Pairwise Test: Mann–Whitney U Test')

                    # mann-whitney U test
                    post_hoc_results = pg.pairwise_tests(data=data,
                                                         dv=self.evaluation_field, 
                                                         between=CLUSTER_FIELD_NAME_STR, 
                                                         alpha=0.05, 
                                                         padjust='bonf', 
                                                         parametric=False,
                                                         alternative=alternative, 
                                                         return_desc=True)
                    post_hoc_results = post_hoc_results.sort_values(by='p-corr')

                elif omnibus_test_type == CLUSTERING_CENTRAL_TENDENCY_TEST_METHOD_WELCH_ANOVA_STR:

                    print(f'Post-Hoc Pairwise Test: Welch t-Test')

                    # welch t-test
                    post_hoc_results = pg.pairwise_tests(data=data,
                                                         dv=self.evaluation_field, 
                                                         between=CLUSTER_FIELD_NAME_STR, 
                                                         alpha=0.05, 
                                                         padjust='bonf', 
                                                         parametric=True,
                                                         correction=True, 
                                                         alternative=alternative, 
                                                         return_desc=True)
                    post_hoc_results = post_hoc_results.sort_values(by='p-corr')

                else:

                    print(f'Post-Hoc Pairwise Test: t-Test')

                    # t-test
                    post_hoc_results = pg.pairwise_tests(data=data,
                                                         dv=self.evaluation_field, 
                                                         between=CLUSTER_FIELD_NAME_STR, 
                                                         alpha=0.05, 
                                                         padjust='bonf', 
                                                         parametric=True,
                                                         correction=False, 
                                                         alternative=alternative, 
                                                         return_desc=True)
                    post_hoc_results = post_hoc_results.sort_values(by='p-corr')
                
                return post_hoc_results
            
            else:
                raise Exception(f'Group {group_str} has only one cluster. Post-Hoc Tests are not possible.')
        
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def aggregate_sequence_clustering_and_eval_metric_diff_test_over_groups(self):
        """Aggregates the clustering results over all groups and inter alia calculates the percentage of groups having\
        significant differences in central tendency of the evaluation metric between clusters.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            # calculate result fields
            n_groups = self.cluster_eval_metric_central_tendency_differences_per_group.shape[0]
            n_group_multiple_clusters = sum(self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR] > 1)
            aggregates = self.cluster_eval_metric_central_tendency_differences_per_group\
                            .groupby([CLUSTERING_DATASET_NAME_FIELD_NAME_STR])\
                            [[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR, 
                            CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR, 
                            CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]]\
                            .agg([np.mean, np.median, min, max, np.std, iqr])
            aggregates_multiple_clusters = self.cluster_eval_metric_central_tendency_differences_per_group\
                                                .loc[self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR] > 1, :]\
                                                .groupby([CLUSTERING_DATASET_NAME_FIELD_NAME_STR])\
                                                [[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR, 
                                                CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR, 
                                                CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]]\
                                                .agg([np.mean, np.median, min, max, np.std, iqr])
            n_groups_significant_cluster_central_tendencies_differences = sum(self.cluster_eval_metric_central_tendency_differences_per_group[CLUSTERING_CENTRAL_TENDENCY_DIFFERENCES_TEST_PVAL_FIELD_NAME_STR] < 0.05)   
            percentage_groups_significant_cluster_central_tendencies_differences = n_groups_significant_cluster_central_tendencies_differences / n_groups * 100
            percentage_groups_with_multiple_clusters_significant_cluster_central_tendencies_differences = n_groups_significant_cluster_central_tendencies_differences / n_group_multiple_clusters * 100

            # fill the result dictionary
            results_dict = {CLUSTERING_NUMBER_OF_GROUPS_FIELD_NAME_STR: n_groups,
                            CLUSTERING_NUMBER_OF_GROUPS_WITH_MULTIPLE_CLUSTERS_FIELD_NAME_STR: n_group_multiple_clusters,
                            CLUSTERING_MEAN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_MEAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_MEAN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_CLUSTERS_PER_GROUP_FIELD_NAME_STR: aggregates[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_MEAN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_SEQUENCES_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_MEAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_UNIQUE_SEQUENCES_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_UNIQUE_SEQUENCES_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_MEAN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['mean'],
                            CLUSTERING_MEDIAN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['median'],
                            CLUSTERING_MIN_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['min'],
                            CLUSTERING_MAX_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['max'],
                            CLUSTERING_STD_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['std'],
                            CLUSTERING_IQR_NUMBER_CLUSTERS_PER_GROUP_MULTIPLE_CLUSTERS_FIELD_NAME_STR: aggregates_multiple_clusters[CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR]['iqr'],
                            CLUSTERING_NUMBER_OF_GROUPS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: n_groups_significant_cluster_central_tendencies_differences,
                            CLUSTERING_PCT_OF_GROUPS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: percentage_groups_significant_cluster_central_tendencies_differences,
                            CLUSTERING_PCT_OF_GROUPS_WITH_MULTIPLE_CLUSTERS_SIG_DIFF_EVAL_METRIC_CENTRAL_TENDENCIES_BETWEEN_CLUSTERES_FIELD_NAME_STR: percentage_groups_with_multiple_clusters_significant_cluster_central_tendencies_differences}

            # result dataframe -> instance attribute
            self.aggregate_sequence_clustering_and_eval_metric_diff_test = pd.DataFrame(results_dict)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_clusters_by_group_umap(self,
                                       group_str: str,
                                       **kwargs):
        """Reduces the sequence distance matrix of specified group with UMAP and displays clustering in a two-dimensional projection.

        Parameters
        ----------
        group_str : str
            A string indicating for which group the sequence distance clustering will be displayed
        **kwargs
            Keyword arguments for the UMAP class. Random_state and verbose are preset.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            reducer = umap.UMAP(random_state=1,
                                verbose = False,
                                **kwargs)

            embedding_2D = reducer.fit_transform(self._square_matrix_per_group[group_str])

            clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group_str].values()), dtype=int)[self._clustered_per_group[group_str]]

            g = sns.scatterplot(x = embedding_2D[~self._clustered_per_group[group_str], 0], y = embedding_2D[~self._clustered_per_group[group_str], 1], s=100, marker=".", alpha =1, color="black")
            g = sns.scatterplot(x = embedding_2D[self._clustered_per_group[group_str], 0], y = embedding_2D[self._clustered_per_group[group_str], 1], s=100, hue=clustered_labels, marker=".", alpha =0.5, palette=sns.husl_palette(len(np.unique(clustered_labels))))
            g.set(xlabel='Embedding 1',
                ylabel='Embedding 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title = "Clusters", title_fontsize = 20);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_clusters_by_group_pca(self,
                                      group_str: str):
        """Reduces the sequence distance matrix of specified group with PCA and displays clustering in a two-dimensional projection.

        Parameters
        ----------
        group_str : str
            A string indicating for which group the sequence distance clustering will be displayed

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            reducer = PCA(n_components=2)
            embedding_2D = reducer.fit_transform(self._square_matrix_per_group[group_str])

            clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group_str].values()), dtype=int)[self._clustered_per_group[group_str]]

            g = sns.scatterplot(x = embedding_2D[~self._clustered_per_group[group_str], 0], y = embedding_2D[~self._clustered_per_group[group_str], 1], s=100, marker=".", alpha =1, color="black")
            g = sns.scatterplot(x = embedding_2D[self._clustered_per_group[group_str], 0], y = embedding_2D[self._clustered_per_group[group_str], 1], s=100, hue=clustered_labels, marker=".", alpha =0.5, palette=sns.husl_palette(len(np.unique(clustered_labels))))
            g.set(xlabel='Component 1',
                ylabel='Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title = "Clusters", title_fontsize = 20);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_eval_metric_dist_between_clusters_by_group(self,
                                                           group_str: str):
        """Displays the evaluation metric distribution per cluster via boxplots, violinplot, boxenplot and kde for specified group.

        Parameters
        ----------
        group_str : str
            A string indicating for which group the evaluation metric distribution per cluster will be displayed

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            g = sns.boxplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field, 
                            showmeans=True, 
                            meanprops=marker_config,
                            showfliers=False)
            g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field, 
                            size=2, 
                            color="red")
            plt.show()

            g = sns.violinplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field,
                            showmeans=True, 
                            meanprops=marker_config,
                            showfliers=False)
            g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field, 
                            size=2, 
                            color="red")
            plt.show()

            g = sns.boxenplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR,
                            y=self.evaluation_field,
                            showfliers=False)
            g = sns.stripplot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field, 
                            size=2, 
                            color="red")
            plt.show()

            g = sns.displot(data=self._user_cluster_eval_metric_df_per_group[group_str], 
                            x=self.evaluation_field, 
                            hue=CLUSTER_FIELD_NAME_STR, 
                            kind='kde')
            plt.show()

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')
    

    def display_cluster_size_by_group(self,
                                      group_str: str):
        """Displays the cluster sizes for specified group

        Parameters
        ----------
        group_str : str
            A string indicating for which group the evaluation metric distribution per cluster will be displayed

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTER_FIELD_NAME_STR} Size for {GROUP_FIELD_NAME_STR} {group_str}:')
            print('-'*100)
            g = sns.barplot(data=self.cluster_stats_per_group.loc[self.cluster_stats_per_group[GROUP_FIELD_NAME_STR]==group_str, :], 
                            x=CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
                            y=CLUSTER_FIELD_NAME_STR)
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')
    
    def display_eval_metric_dist_between_cluster_all_groups(self,
                                                            height: int):
        """Displays the evaluation metric distribution per cluster via boxplots for all groups.

        Parameters
        ----------
        height : int
            The height of the subplots

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """

        if self._data_clustered:        
            # concat user_cluster_eval_metric_df over all groups
            user_cluster_eval_metric_df = pd.DataFrame()            
            for k,v in self._user_cluster_eval_metric_df_per_group.items():
                v[GROUP_FIELD_NAME_STR] = k
                user_cluster_eval_metric_df = pd.concat([user_cluster_eval_metric_df, v])
            
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'Central Tendency Differences in {CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR} between {CLUSTER_FIELD_NAME_STR}s per {GROUP_FIELD_NAME_STR}:')
            print(f'Chosen {CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR}: "{self.evaluation_field}"')
            print('-'*100)
            g = sns.FacetGrid(user_cluster_eval_metric_df, 
                              col=GROUP_FIELD_NAME_STR, 
                              col_wrap=6, 
                              sharex=False,
                              sharey=False,
                              height=height, 
                              aspect= 1)
            g.map_dataframe(sns.boxplot, 
                            x=CLUSTER_FIELD_NAME_STR, 
                            y=self.evaluation_field,
                            showmeans=True, 
                            meanprops=marker_config_eval_metric_mean)
            g.set(xlabel=CLUSTER_FIELD_NAME_STR, 
                  ylabel=CLUSTERING_EVALUATION_METRIC_FIELD_NAME_STR)
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            g.fig.subplots_adjust(top=0.95)
            g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_clusters_all_group_umap(self,
                                        height: int,
                                        **kwargs):
        """Reduces the sequence distance matrices of all group with UMAP and displays clustering in a two-dimensional projection.

        Parameters
        ----------
        height : int
            The height of the subplots
        **kwargs
            Keyword arguments for the UMAP class. Random_state and verbose are preset.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            reducer = umap.UMAP(random_state=1,
                                verbose = False,
                                **kwargs)

            emb_per_group_df = pd.DataFrame()

            for group, distance_matrix in self._square_matrix_per_group.items():

                embedding_2D = reducer.fit_transform(distance_matrix)

                clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group].values()), dtype=int)[self._clustered_per_group[group]]

                emb_clustered = embedding_2D[self._clustered_per_group[group], :]
                emb_not_clustered = embedding_2D[~self._clustered_per_group[group], :]
                
                emb_clustered_df = pd.DataFrame(emb_clustered, columns=['x_clust', 'y_clust'])
                emb_clustered_df[CLUSTER_FIELD_NAME_STR] = clustered_labels
                emb_not_clustered_df = pd.DataFrame(emb_not_clustered, columns=['x_not_clust', 'y_not_clust'])
                emb_df = pd.concat([emb_clustered_df, emb_not_clustered_df], ignore_index=False, axis=1)
                emb_df[GROUP_FIELD_NAME_STR] = group

                emb_per_group_df = pd.concat([emb_per_group_df, emb_df])

            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTER_FIELD_NAME_STR}s per Group')
            print(f'Dimensionality Reducer: UMAP')
            print('-'*100)
            g = sns.FacetGrid(emb_per_group_df, 
                              col=GROUP_FIELD_NAME_STR, 
                              col_wrap=6, 
                              sharex=False,
                              sharey=False,
                              height=height, 
                              aspect= 1)
            g.map_dataframe(sns.scatterplot, 
                            x='x_not_clust', 
                            y='y_not_clust',
                            color='black',
                            alpha=1,
                            s=10)
            g.map_dataframe(sns.scatterplot, 
                            x='x_clust', 
                            y='y_clust',
                            hue=CLUSTER_FIELD_NAME_STR,
                            palette=sns.color_palette('deep', as_cmap=False),
                            alpha=1,
                            s=10)
            g.set(xlabel='Embedding 1', 
                  ylabel='Embedding 2')
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            g.fig.subplots_adjust(top=0.95)
            g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_clusters_all_group_pca(self,
                                       height: int):
        """Reduces the sequence distance matrices of all group with PCA and displays clustering in a two-dimensional projection.

        Parameters
        ----------
        height : int
            The height of the subplots

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            reducer = PCA(n_components=2)

            emb_per_group_df = pd.DataFrame()

            for group, distance_matrix in self._square_matrix_per_group.items():

                embedding_2D = reducer.fit_transform(distance_matrix)

                clustered_labels = np.array(list(self._user_cluster_mapping_per_group[group].values()), dtype=int)[self._clustered_per_group[group]]

                emb_clustered = embedding_2D[self._clustered_per_group[group], :]
                emb_not_clustered = embedding_2D[~self._clustered_per_group[group], :]
                
                emb_clustered_df = pd.DataFrame(emb_clustered, columns=['x_clust', 'y_clust'])
                emb_clustered_df[CLUSTER_FIELD_NAME_STR] = clustered_labels
                emb_not_clustered_df = pd.DataFrame(emb_not_clustered, columns=['x_not_clust', 'y_not_clust'])
                emb_df = pd.concat([emb_clustered_df, emb_not_clustered_df], ignore_index=False, axis=1)
                emb_df[GROUP_FIELD_NAME_STR] = group

                emb_per_group_df = pd.concat([emb_per_group_df, emb_df])

            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTER_FIELD_NAME_STR}s per Group')
            print(f'Dimensionality Reducer: PCA')
            print('-'*100)
            g = sns.FacetGrid(emb_per_group_df, 
                              col=GROUP_FIELD_NAME_STR, 
                              col_wrap=6, 
                              sharex=False,
                              height=height, aspect= 1)
            g.map_dataframe(sns.scatterplot, 
                            x='x_not_clust', 
                            y='y_not_clust',
                            color='black',
                            alpha=1,
                            s=10)
            g.map_dataframe(sns.scatterplot, 
                            x='x_clust', 
                            y='y_clust',
                            hue=CLUSTER_FIELD_NAME_STR,
                            palette=sns.color_palette('deep', as_cmap=False),
                            alpha=1,
                            s=10)
            g.set(xlabel='Embedding 1', 
                  ylabel='Embedding 2')
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            g.fig.subplots_adjust(top=0.95)
            g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_number_of_clusters_all_group(self):
        """Displays the number of clusters per group as barplot.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
            print('-'*100)
            g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
                            x=CLUSTERING_NUMBER_CLUSTERS_FIELD_NAME_STR,
                            y=GROUP_FIELD_NAME_STR)
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_min_cluster_size_all_group(self):
        """Displays the size of the smallest cluster per group as barplot.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
            print('-'*100)
            g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
                            x=CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR,
                            y=GROUP_FIELD_NAME_STR)
            plt.show(g)
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_percentage_clustered_all_group(self):
        """Displays the percentage of sequencese clustered per group as barplot.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
            print('-'*100)
            g = sns.barplot(data=self.cluster_eval_metric_central_tendency_differences_per_group,
                            x=CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR,
                            y=GROUP_FIELD_NAME_STR)
            plt.show(g)
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def display_cluster_size_all_group(self,
                                       height: int):
        """Displays the cluster sizes for each group as multi-grid barplot.

        Parameters
        ----------
        height : int
            The height of the subplots

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTER_FIELD_NAME_STR} Size per {GROUP_FIELD_NAME_STR}:')
            print(f'{CLUSTER_FIELD_NAME_STR} Size -> Number of {USER_FIELD_NAME_STR} {LEARNING_ACTIVITY_FIELD_NAME_STR}-{SEQUENCE_STR}s')
            print('-'*100)
            print(f'Plots:')
            g = sns.FacetGrid(self.cluster_stats_per_group, 
                              col=GROUP_FIELD_NAME_STR, 
                              col_wrap=6, 
                              sharex=False,
                              sharey=False,
                              height=height, 
                              aspect= 1)
            g.map_dataframe(sns.barplot, 
                            x=CLUSTERING_NUMBER_SEQUENCES_PER_CLUSTER_FIELD_NAME_STR, 
                            y=CLUSTER_FIELD_NAME_STR)
            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)
            g.fig.subplots_adjust(top=0.95)
            g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]);
            plt.show(g)

        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def print_min_cluster_sizes_all_group(self):
        """Print the minimum cluster sizes sizes for each group.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
            print('-'*100)
            print(self.cluster_eval_metric_central_tendency_differences_per_group[[GROUP_FIELD_NAME_STR, CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR]]\
                  .sort_values(by=CLUSTERING_MIN_CLUSTER_SIZE_FIELD_NAME_STR)\
                  .to_string(index=False))
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')

    def print_percentage_clustered_all_group(self):
        """Print the percentage of sequences clustered for each group.

        Raises
        ------
        Exception
            Clustering via cluster_sequences_and_test_eval_metric_diff method needs to be performed first in order to use this method
        """
        if self._data_clustered:        
            print('*'*100)
            print('*'*100)
            print(' ')
            print('-'*100)
            print(f'{CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR} per {GROUP_FIELD_NAME_STR}:')
            print('-'*100)
            print(self.cluster_eval_metric_central_tendency_differences_per_group[[GROUP_FIELD_NAME_STR, CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR]]\
                  .sort_values(by=CLUSTERING_PERCENTAGE_CLUSTERED_FIELD_NAME_STR)\
                  .to_string(index=False))
        else:
            raise Exception(f'Method not applicable. Please apply "cluster_sequences_and_test_eval_metric_diff" method first!')