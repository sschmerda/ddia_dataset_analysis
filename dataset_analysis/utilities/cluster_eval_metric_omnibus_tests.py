from .standard_import import *
from .constants import *
from .config import *
from .plotting_functions import *
from .validators import *
from .data_classes import *
from .omnibus_tests_config import *

class ClusterEvalMetricOmnibusTest():
    """docstring for ClassName."""

    def __init__(self, 
                 dataset_name: str,
                 interactions: pd.DataFrame,
                 sequence_cluster_per_group_df: pd.DataFrame,
                 evaluation_metric_field: str,
                 evaluation_metric_is_categorical: bool,
                 exclude_non_clustered: bool) -> None:

        self.dataset_name = dataset_name
        self.interactions = interactions
        self.sequence_cluster_per_group_df = sequence_cluster_per_group_df
        self.evaluation_metric_field = evaluation_metric_field
        self.evaluation_metric_field_is_categorical = evaluation_metric_is_categorical
        self.exclude_non_clustered = exclude_non_clustered

        # data transformation
        self.sequence_cluster_eval_metric_per_group_df = self._return_seq_cluster_eval_metric_per_group_df(self.sequence_cluster_per_group_df,
                                                                                                           self.interactions)
        self.sequence_cluster_eval_metric_per_group_df = self._exclude_non_clustered_sequences(self.sequence_cluster_eval_metric_per_group_df)

        # initialization of test result df
        self.omnibus_test_result_df = pd.DataFrame()

    def perform_omnibus_tests(self):

        if self.evaluation_metric_field_is_categorical:
            self.omnibus_test_result_df = self._return_omnibus_test_result_categorical_var()
        else:
            self.omnibus_test_result_df = self._return_omnibus_test_result_continuous_var()
    
    def return_omnibus_test_result(self) -> pd.DataFrame:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)
        
        return self.omnibus_test_result_df

    def plot_cluster_eval_metric_per_group_plot(self) -> None:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        if self.evaluation_metric_field_is_categorical:
            self._plot_cluster_eval_metric_mosaic_plot()
        else:
            self._plot_cluster_eval_metric_boxplot()
    
    def _plot_cluster_eval_metric_mosaic_plot(self) -> None:
                                             
        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        test_results_per_group_dict = self._return_test_results_per_group_dict()

        self.sequence_cluster_eval_metric_per_group_df = self.sequence_cluster_eval_metric_per_group_df.sort_values([GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR, self.evaluation_metric_field])

        unique_clusters = self.sequence_cluster_eval_metric_per_group_df[CLUSTER_FIELD_NAME_STR].unique().astype(str)
        unique_eval_metrics = self.sequence_cluster_eval_metric_per_group_df[self.evaluation_metric_field].unique().astype(str)

        palette = return_color_palette(len(unique_clusters))
        color_dict = dict(zip(unique_eval_metrics, palette))

        clust_eval_metr_cartesian_prod = product(unique_clusters, unique_eval_metrics)
        props = {}
        for clust, eval_metric in clust_eval_metr_cartesian_prod:
                props[(clust, eval_metric)] = {'color': color_dict[eval_metric]}

        # helper function
        def plot_mosaic(*args,**kwargs):
            data = kwargs['data']
            cat_vars = list(args)
            g = mosaic(data, 
                       cat_vars,
                       labelizer=lambda k: '',
                       properties=props,
                       gap=0.02,
                       ax=plt.gca())

        n_cols = set_facet_grid_column_number(self.sequence_cluster_eval_metric_per_group_df[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        g = sns.FacetGrid(self.sequence_cluster_eval_metric_per_group_df, 
                          col=GROUP_FIELD_NAME_STR, 
                          col_wrap=n_cols,
                          sharex=False,
                          sharey=False,
                          height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                          aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE)
        g = g.map_dataframe(plot_mosaic, 
                            CLUSTER_FIELD_NAME_STR, 
                            self.evaluation_metric_field)

        axes_iterable = zip(g.axes.flat, g.facet_data())
        for ax, (_, subset) in axes_iterable:

                group = subset[GROUP_FIELD_NAME_STR].iloc[0]
                pval = test_results_per_group_dict[group].pval
                cramersv = test_results_per_group_dict[group].cramersv

                group_str = f'{GROUP_FIELD_NAME_STR} = {group}'
                pvalue_str = f'\n{OMNIBUS_TESTS_CONTINGENCY_PVAL_VALUE_NAME_STR} = {pval}'
                cramersv_str = f'\n{OMNIBUS_TESTS_CONTINGENCY_CRAMERSV_VALUE_NAME_STR} = {cramersv}'
                title_str = group_str + pvalue_str + cramersv_str
                ax.set_title(title_str)

        plt.show(g)

    def _plot_cluster_eval_metric_boxplot() -> None:
        pass

    def _return_omnibus_test_result_categorical_var(self) -> pd.DataFrame:

        small_expected_freq_per_group_df = self._return_number_small_expected_frequencies()

        group_has_small_freq = small_expected_freq_per_group_df[OMNIBUS_TESTS_CONTINGENCY_HAS_SMALL_EXPECTED_FREQ_FIELD_NAME_STR]
        groups_chi_squared = tuple(small_expected_freq_per_group_df.loc[~group_has_small_freq, GROUP_FIELD_NAME_STR])
        groups_fisher = tuple(small_expected_freq_per_group_df.loc[group_has_small_freq, GROUP_FIELD_NAME_STR])

        chi_squared_test_results = self._return_omnibus_test_result_chi2_independence(groups_chi_squared)
        fisher_test_results = self.return_omnibus_test_result_fisher_independence(groups_fisher)

        omnibus_test_results = pd.concat([chi_squared_test_results, fisher_test_results])

        return omnibus_test_results

    def _return_omnibus_test_result_continuous_var(self):
        pass

    def _return_omnibus_test_result_chi2_independence(self,
                                                      groups: tuple[int]) -> pd.DataFrame:

        test_results_per_group_df = pd.DataFrame()

        if len(groups) > 0:
            test_results_df_list = []
            for group, df in self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR):
                if group in groups:

                    crosstab_res = crosstab(df[CLUSTER_FIELD_NAME_STR], df[self.evaluation_metric_field])
                    observed_freq = crosstab_res.count
                    n_observations = np.sum(observed_freq)

                    chi_sq_res = chi2_contingency(observed_freq,
                                                  correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
                    stat = chi_sq_res.statistic
                    pval = chi_sq_res.pvalue
                    dof = chi_sq_res.dof
                    expected_freq = chi_sq_res.expected_freq
                    measure_of_association_type = OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_METHOD_VALUE.value
                    measure_of_association_value = self._return_measure_association_contingency(observed_freq)

                    boot_conf_int_res = self._return_effect_size_conf_interval_bootstrap(df,
                                                                                         self._return_measure_association_contingency_bootstrap)
                    conf_int = boot_conf_int_res.confidence_interval
                    boot_dist = boot_conf_int_res.bootstrap_distribution
                    boot_stde = boot_conf_int_res.standard_error

                    test_results_df = self._return_test_result_df_categorical_var(self.dataset_name,
                                                                                  group,
                                                                                  OMNIBUS_TESTS_CONTINGENCY_TEST_PEARSON_CHI_SQUARED_VALUE_NAME_STR,
                                                                                  OMNIBUS_TESTS_CONTINGENCY_TEST_STATISTIC_CHI_SQUARED_VALUE_NAME_STR,
                                                                                  stat,
                                                                                  dof,
                                                                                  n_observations,
                                                                                  pval,
                                                                                  measure_of_association_type,
                                                                                  measure_of_association_value,
                                                                                  conf_int)

                    test_results_df_list.append(test_results_df)


            test_results_per_group_df = pd.concat(test_results_df_list, 
                                                  ignore_index=True)

        return test_results_per_group_df
    
    def return_omnibus_test_result_fisher_independence(self,
                                                       groups: tuple[int]) -> pd.DataFrame:

        test_results_per_group_df = pd.DataFrame()

        if len(groups) > 0:
            pass

        return test_results_per_group_df

    def _return_number_small_expected_frequencies(self) -> pd.DataFrame:

        if self.evaluation_metric_field_is_categorical:
            group_list = []
            n_expected_freq_list = []
            min_allowed_expected_freq_list = []
            n_small_expected_freq_list = []
            pct_small_expected_freq_list = []
            has_small_expected_freq_list = []

            for group, df in self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR):
                _, observed_freq = crosstab(df[CLUSTER_FIELD_NAME_STR],
                                            df[self.evaluation_metric_field])
                
                expected_freq_matrix = expected_freq(observed_freq)


                expected_freq_matrix = expected_freq_matrix
                n_expected_freq = expected_freq_matrix.size
                has_small_expected_freq_matrix = (expected_freq_matrix < OMNIBUS_TESTS_CONTINGENCY_MIN_EXPECTED_FREQ_VALUE)
                has_small_expected_freq = has_small_expected_freq_matrix.any()
                n_small_expected_freq = has_small_expected_freq_matrix.sum()
                pct_small_expected_freq = n_small_expected_freq / n_expected_freq * 100

                group_list.append(group)
                n_expected_freq_list.append(n_expected_freq)
                min_allowed_expected_freq_list.append(OMNIBUS_TESTS_CONTINGENCY_MIN_EXPECTED_FREQ_VALUE)
                n_small_expected_freq_list.append(n_small_expected_freq)
                pct_small_expected_freq_list.append(pct_small_expected_freq)
                has_small_expected_freq_list.append(has_small_expected_freq)
            
            small_expected_freq_per_group_df = pd.DataFrame({GROUP_FIELD_NAME_STR: group_list,
                                                             OMNIBUS_TESTS_CONTINGENCY_NUMBER_EXPECTED_FREQ_FIELD_NAME_STR: n_expected_freq_list,
                                                             OMNIBUS_TESTS_CONTINGENCY_MIN_ALLOWED_EXPECTED_FREQ_FIELD_NAME_STR: min_allowed_expected_freq_list,
                                                             OMNIBUS_TESTS_CONTINGENCY_NUMBER_SMALL_EXPECTED_FREQ_FIELD_NAME_STR: n_small_expected_freq_list,
                                                             OMNIBUS_TESTS_CONTINGENCY_PERCENTAGE_SMALL_EXPECTED_FREQ_FIELD_NAME_STR: pct_small_expected_freq_list,
                                                             OMNIBUS_TESTS_CONTINGENCY_HAS_SMALL_EXPECTED_FREQ_FIELD_NAME_STR: has_small_expected_freq_list})
            
            return small_expected_freq_per_group_df
        
        raise ValueError(OMNIBUS_TESTS_ERROR_EVAL_METRIC_NOT_CATEGORICAL_NAME_STR)
    
    def _return_measure_association_contingency(self,
                                                observed_freq: np.ndarray):
        measure_of_association = association(observed_freq,
                                             method=OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_METHOD_VALUE.name,
                                             correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
        return measure_of_association

    @staticmethod
    def _return_measure_association_contingency_bootstrap(x, y):
        crosstab_res = crosstab(x, y)
        observed_freq = crosstab_res.count
        measure_of_association = ClusterEvalMetricOmnibusTest._return_measure_association_contingency(observed_freq)
                            
        return measure_of_association

    def _return_effect_size_conf_interval_bootstrap(self,
                                                    sequence_cluster_eval_metric_df: pd.DataFrame,
                                                    effect_size_func: Callable):

        boot_res = bootstrap((sequence_cluster_eval_metric_df[CLUSTER_FIELD_NAME_STR].values, sequence_cluster_eval_metric_df[self.evaluation_metric_field].values),
                              effect_size_func,
                              n_resamples=OMNIBUS_TESTS_BOOTSTRAPPING_N_RESAMPLES,
                              vectorized=OMNIBUS_TESTS_BOOTSTRAPPING_VECTORIZED,
                              paired=OMNIBUS_TESTS_BOOTSTRAPPING_PAIRED,
                              confidence_level=OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                              alternative=OMNIBUS_TESTS_BOOTSTRAPPING_ALTERNATIVE,
                              method=OMNIBUS_TESTS_BOOTSTRAPPING_METHOD,
                              random_state=RNG_SEED)

        conf_int = tuple(boot_res.confidence_interval)
        boot_dist = boot_res.bootstrap_distribution
        boot_stde = boot_res.standard_error

        return conf_int
    
    def _return_test_result_df_categorical_var(self,
                                               dataset_name: str,
                                               group: int,
                                               test_type: str,
                                               test_statistic_type: str | None,
                                               test_statistic_value: float | None,
                                               degrees_of_freedom: int | None,
                                               number_observations: int,
                                               p_val: float,
                                               measure_of_association_type: str,
                                               measure_of_association_value: float,
                                               measure_of_association_conf_int: tuple[float, float],
                                               measure_of_association_conf_int_level: float):

        test_results_df = pd.DataFrame({DATASET_NAME_FIELD_NAME_STR: dataset_name,
                                        GROUP_FIELD_NAME_STR: group,
                                        OMNIBUS_TESTS_CONTINGENCY_TEST_TYPE_FIELD_NAME_STR: test_type,
                                        OMNIBUS_TESTS_CONTINGENCY_TEST_STATISTIC_TYPE_FIELD_NAME_STR: test_statistic_type,
                                        OMNIBUS_TESTS_CONTINGENCY_TEST_STATISTIC_VALUE_FIELD_NAME_STR: test_statistic_value,
                                        OMNIBUS_TESTS_CONTINGENCY_DEGREES_OF_FREEDOM_FIELD_NAME_STR: degrees_of_freedom,
                                        OMNIBUS_TESTS_CONTINGENCY_SAMPLE_SIZE_FIELD_NAME_STR: number_observations,
                                        OMNIBUS_TESTS_CONTINGENCY_PVAL_FIELD_NAME_STR: p_val,
                                        OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_TYPE_FIELD_NAME_STR: measure_of_association_type,
                                        OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_VALUE_FIELD_NAME_STR: measure_of_association_value,
                                        OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_CONF_INT_FIELD_NAME_STR: measure_of_association_conf_int,
                                        OMNIBUS_TESTS_CONTINGENCY_EFFECT_SIZE_CONF_INT_ALPHA_FIELD_NAME_STR: measure_of_association_conf_int_level})

        return test_results_df
    
    def _return_seq_cluster_eval_metric_per_group_df(self,
                                                     sequence_cluster_per_group_df: pd.DataFrame,
                                                     interactions: pd.DataFrame) -> pd.DataFrame:

        group_user_eval_metric_fields = [GROUP_FIELD_NAME_STR, USER_FIELD_NAME_STR]
        group_user_seq_eval_metric_df = (interactions.groupby(group_user_eval_metric_fields)
                                         [self.evaluation_metric_field].first().reset_index())

        sequence_cluster_per_group_df = pd.merge(self.sequence_cluster_per_group_df, 
                                                 group_user_seq_eval_metric_df,
                                                 how='left',
                                                 on=group_user_eval_metric_fields)
        return sequence_cluster_per_group_df
    
    def _exclude_non_clustered_sequences(self,
                                         data: pd.DataFrame) -> pd.DataFrame:
        if self.exclude_non_clustered:
            data = data.query(f'{CLUSTER_FIELD_NAME_STR}!=-1')
        
        return data

    def _return_test_results_per_group_dict(self) -> dict[int, TestResultsContingency | TestResultsContinuous]:

        #TODO: adapt to new constants
        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        test_results_per_group_dict = {}
        for group, df in self.omnibus_test_result_df.groupby(GROUP_FIELD_NAME_STR):

            if self.evaluation_metric_field_is_categorical:
                test_results = TestResultsContingency(group,
                                                      df[OMNIBUS_TESTS_CONTINGENCY_TEST_FIELD_NAME_STR].values[0],
                                                      df[OMNIBUS_TESTS_CONTINGENCY_TEST_STATISTIC_FIELD_NAME_STR].values[0],
                                                      df[OMNIBUS_TESTS_CONTINGENCY_DEGREES_OF_FREEDOM_FIELD_NAME_STR].values[0],
                                                      df[OMNIBUS_TESTS_CONTINGENCY_PVAL_FIELD_NAME_STR].values[0],
                                                      df[OMNIBUS_TESTS_CONTINGENCY_CRAMERSV_FIELD_NAME_STR].values[0],
                                                      (0.0, 0.0),
                                                      df[OMNIBUS_TESTS_CONTINGENCY_POWER_FIELD_NAME_STR])


            else:
                test_results = TestResultsContinuous(group,
                                                     df[OMNIBUS_TESTS_CONTINGENCY_TEST_FIELD_NAME_STR],
                                                     df[OMNIBUS_TESTS_CONTINGENCY_TEST_STATISTIC_FIELD_NAME_STR],
                                                     df[OMNIBUS_TESTS_CONTINGENCY_DEGREES_OF_FREEDOM_FIELD_NAME_STR],
                                                     df[OMNIBUS_TESTS_CONTINGENCY_PVAL_FIELD_NAME_STR],
                                                     df[OMNIBUS_TESTS_CONTINGENCY_CRAMERSV_FIELD_NAME_STR],
                                                     (0.0, 0.0),
                                                     df[OMNIBUS_TESTS_CONTINGENCY_POWER_FIELD_NAME_STR])

            test_results_per_group_dict[group] = test_results


        return test_results_per_group_dict