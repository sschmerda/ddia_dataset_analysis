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
                 exclude_non_clustered: bool,
                 include_r_test_results: bool) -> None:

        self.dataset_name: str = dataset_name
        self.interactions: pd.DataFrame = interactions
        self.sequence_cluster_per_group_df: pd.DataFrame = sequence_cluster_per_group_df
        self.evaluation_metric_field: str = evaluation_metric_field
        self.evaluation_metric_field_is_categorical: bool = evaluation_metric_is_categorical
        self.exclude_non_clustered: bool = exclude_non_clustered
        self.include_r_test_results: bool = include_r_test_results
        self._measure_association_fail_dict_per_group: DefaultDict[int, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._measure_of_association_results_per_group: dict[int, list[MeasureAssociationContingencyResults | MeasureAssociationAOVResults]] = {}

        # data transformation
        self.sequence_cluster_eval_metric_per_group_df = self._return_seq_cluster_eval_metric_per_group_df(self.sequence_cluster_per_group_df,
                                                                                                           self.interactions)
        self.sequence_cluster_eval_metric_per_group_df = self._exclude_non_clustered_sequences(self.sequence_cluster_eval_metric_per_group_df)

        # initialization of test result df
        self.omnibus_test_result_df: pd.DataFrame = pd.DataFrame()

    def perform_omnibus_tests(self) -> None:

        if self.evaluation_metric_field_is_categorical:
            self.omnibus_test_result_df = self._return_omnibus_test_result_categorical_var()
        else:
            self.omnibus_test_result_df = self._return_omnibus_test_result_continuous_var()
    
    def return_omnibus_test_result(self) -> pd.DataFrame:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)
        
        return self.omnibus_test_result_df

    def return_measure_association_conf_int_bootstrap_failures(self) -> pd.DataFrame:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        moa_fail_dict_list = []
        for group, moa_dict in self._measure_association_fail_dict_per_group.items():
            moa_fail_dict = {GROUP_FIELD_NAME_STR: group}
            for moa, fail_count in moa_dict.items():

                fail_count_pct = (fail_count / OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES * 100)

                moa_fail_count_field_name = moa + OMNIBUS_TESTS_MEASURE_ASSOCIATION_BOOTSTRAP_CONF_INT_FAIL_COUNT_FIELD_NAME_STR
                moa_fail_pct_field_name = moa + OMNIBUS_TESTS_MEASURE_ASSOCIATION_BOOTSTRAP_CONF_INT_FAIL_PCT_FIELD_NAME_STR

                moa_fail_dict[moa_fail_count_field_name] = fail_count
                moa_fail_dict[moa_fail_pct_field_name] = fail_count_pct

            moa_fail_dict_list.append(moa_fail_dict)

        measure_association_conf_int_fail_count = pd.DataFrame(moa_fail_dict_list)
        measure_association_conf_int_fail_count

        return measure_association_conf_int_fail_count 
    
    def return_measure_association_results_per_group_dict(self) -> dict:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        return self._measure_of_association_results_per_group

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

        self.sequence_cluster_eval_metric_per_group_df = self.sequence_cluster_eval_metric_per_group_df.sort_values([GROUP_FIELD_NAME_STR, CLUSTER_FIELD_NAME_STR, self.evaluation_metric_field])

        unique_clusters = self.sequence_cluster_eval_metric_per_group_df[CLUSTER_FIELD_NAME_STR].unique().astype(str)
        unique_eval_metrics = self.sequence_cluster_eval_metric_per_group_df[self.evaluation_metric_field].unique().astype(str)

        palette = return_color_palette(len(unique_eval_metrics))
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
        
        self.add_plot_data_title(g,
                                 OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE,
                                 FACET_GRID_SUBPLOTS_H_SPACE_SQUARE_WITH_TITLE)

        plt.show(g)

    def _plot_cluster_eval_metric_boxplot(self) -> None:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        n_cols = set_facet_grid_column_number(self.sequence_cluster_eval_metric_per_group_df[GROUP_FIELD_NAME_STR],
                                              SEABORN_SEQUENCE_FILTER_FACET_GRID_N_COLUMNS)

        g = sns.FacetGrid(self.sequence_cluster_eval_metric_per_group_df,
                          col=GROUP_FIELD_NAME_STR,
                          col_wrap=n_cols, 
                          sharex=False, 
                          sharey=False,
                          height=SEABORN_FIGURE_LEVEL_HEIGHT_SQUARE_FACET,
                          aspect=SEABORN_FIGURE_LEVEL_ASPECT_SQUARE)

        g.map_dataframe(self._draw_variable_width_boxplot, 
                        x=CLUSTER_FIELD_NAME_STR,
                        y=self.evaluation_metric_field,
                        orient='v')

        self.add_plot_data_title(g,
                                 OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE,
                                 FACET_GRID_SUBPLOTS_H_SPACE_SQUARE_WITH_TITLE)

        plt.show(g)

    def _draw_variable_width_boxplot(*args,
                                     **kwargs) -> None:
        data = kwargs.pop('data')

        widths = data[CLUSTER_FIELD_NAME_STR].value_counts(normalize=False)
        widths /= widths.max() * 1.2
        widths = widths.sort_index()

        sns.boxplot(data,
                    **kwargs,
                    widths=widths,
                    showmeans=True, 
                    meanprops=marker_config_eval_metric_mean)
        
    def add_plot_data_title(self,
                            sns_plot,
                            measure_of_association_method: ContingencyEffectSizeEnum | AOVEffectSizeEnum,
                            h_space_title: int | float) -> None:

        axes_iterable = zip(sns_plot.axes.flat, sns_plot.facet_data())
        for ax, (_, subset) in axes_iterable:

            measure_of_association_value_field_name = measure_of_association_method.value + OMNIBUS_TESTS_EFFECT_SIZE_VALUE_FIELD_NAME_STR
            measure_of_association_conf_int_value_field_name = measure_of_association_method.value + OMNIBUS_TESTS_EFFECT_SIZE_CONF_INT_VALUE_FIELD_NAME_STR

            if self.evaluation_metric_field_is_categorical:
                n_obs_field = OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR
                has_exp_freq_below_t_field = OMNIBUS_TESTS_CONTINGENCY_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_FIELD_NAME_STR
                test_statistic_field = OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_TEST_STATISTIC_FIELD_NAME_STR
                p_val_perm_field = OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_PERM_PVAL_FIELD_NAME_STR
            else:
                n_obs_field = OMNIBUS_TESTS_CONTINUOUS_AOV_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR
                test_statistic_field = OMNIBUS_TESTS_CONTINUOUS_AOV_F_TEST_STATISTIC_FIELD_NAME_STR
                p_val_perm_field = OMNIBUS_TESTS_CONTINUOUS_AOV_PERM_PVAL_FIELD_NAME_STR

            # extract data for plotting
            group = subset[GROUP_FIELD_NAME_STR].iloc[0]
            is_group_series = self.omnibus_test_result_df[GROUP_FIELD_NAME_STR] == group
            n_observations = self.omnibus_test_result_df.loc[is_group_series, n_obs_field].values[0] 
            if self.evaluation_metric_field_is_categorical:
                has_expected_frequency_below_threshold = self.omnibus_test_result_df.loc[is_group_series, has_exp_freq_below_t_field].values[0] 
            test_statistic = round(self.omnibus_test_result_df.loc[is_group_series, test_statistic_field].values[0], 3)
            p_value_perm = round(self.omnibus_test_result_df.loc[is_group_series, p_val_perm_field].values[0], 3)
            measure_of_association_type = measure_of_association_value_field_name
            measure_of_association_value = round(self.omnibus_test_result_df.loc[is_group_series, measure_of_association_value_field_name].values[0], 3)
            measure_of_association_conf_int = self.omnibus_test_result_df.loc[is_group_series, measure_of_association_conf_int_value_field_name].values[0]
            measure_of_association_conf_int = tuple(map(lambda x: round(x, 3), measure_of_association_conf_int))
            measure_of_association_conf_int_lvl = int(OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL * 100)
    
            # generate title strings for plot
            if p_value_perm > 0.05:
                star_str = ''
            elif 0.01 < p_value_perm <= 0.05:
                star_str = '*'
            elif 0.001 < p_value_perm <= 0.01:
                star_str = '**'
            else:
                star_str = '***'

            group_str = f'{GROUP_FIELD_NAME_STR}: {group}'
            n_observations_str = f'\n{OMNIBUS_TESTS_NUMBER_OBSERVATIONS_PLOT_VALUE_NAME_STR}: {n_observations}'
            if self.evaluation_metric_field_is_categorical:
                has_expected_frequency_below_threshold_str = f'\n{OMNIBUS_TESTS_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_PLOT_VALUE_NAME_STR}{OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE}: {has_expected_frequency_below_threshold}'
            else:
                has_expected_frequency_below_threshold_str = ''
            test_statistic_str = f'\n{OMNIBUS_TESTS_CHI_SQUARED_PLOT_VALUE_NAME_STR}: {test_statistic}'

            if p_value_perm <= 0.05:
                pvalue_star_str = f'$\mathbf{ {p_value_perm} }$ $\mathbf{ {star_str} }$'
            else:
                pvalue_star_str = f'{p_value_perm}{star_str}'
            pvalue_perm_str = f'\n{OMNIBUS_TESTS_PVAL_PERM_PLOT_VALUE_NAME_STR}: ' + pvalue_star_str

            sub_strings = measure_of_association_type.split('_')
            measure_of_association_type = '_'.join([sub_str.capitalize() for sub_str in sub_strings])
            measure_of_association_str = f'\n{measure_of_association_type}: {measure_of_association_value}'
            measure_of_association_conf_int_str = f'\n{measure_of_association_conf_int_lvl}% {OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_PLOT_VALUE_NAME_STR}: {measure_of_association_conf_int}'

            title_str = ''.join((group_str,
                                 n_observations_str,
                                 has_expected_frequency_below_threshold_str,
                                 test_statistic_str,
                                 pvalue_perm_str,
                                 measure_of_association_str,
                                 measure_of_association_conf_int_str))
            ax.set_title(title_str)
        
        plt.subplots_adjust(hspace=h_space_title)

    def _return_omnibus_test_result_categorical_var(self) -> pd.DataFrame:

        omnibus_test_results = self._return_omnibus_test_result_chi_squared_independence()

        return omnibus_test_results

    def _return_omnibus_test_result_continuous_var(self) -> pd.DataFrame:

        omnibus_test_results = self._return_omnibus_test_result_aov()

        return omnibus_test_results

    def _return_omnibus_test_result_chi_squared_independence(self) -> pd.DataFrame:

        test_results_per_group_df = pd.DataFrame()

        test_results_df_list = []
        for group, df in self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR):

            clusters = df[CLUSTER_FIELD_NAME_STR].values
            eval_metrics = df[self.evaluation_metric_field].values

            # get chi squared test results which includes the chi squared statistic
            chi_squared_test_results = self._return_chi_squared_test_results(clusters,
                                                                             eval_metrics,
                                                                             False,
                                                                             None)

            expected_freq_stats = self._return_expected_frequencies_stats(chi_squared_test_results.expected_frequency)

            # perform a chi squared permutation test
            p_value_perm = self._return_chi_squared_perm_p_value(clusters,
                                                                 eval_metrics,
                                                                 chi_squared_test_results)

            # perform the chi squared test in R as sanity check
            if self.include_r_test_results:
                p_value_r = self._return_chi_squared_p_value_r(clusters,
                                                               eval_metrics,
                                                               False)
                p_value_r_perm = self._return_chi_squared_p_value_r(clusters,
                                                                    eval_metrics,
                                                                    True)
            else:
                p_value_r, p_value_r_perm = None, None

            # calculate measure of association results
            measure_of_association_contingency_results_list = self._return_measure_association_contingency_results(clusters,
                                                                                                                   eval_metrics,
                                                                                                                   chi_squared_test_results.observed_frequency,
                                                                                                                   group)
            self._measure_of_association_results_per_group[group] = measure_of_association_contingency_results_list

            test_results_df = self._return_test_result_df_chi_squared_independence(self.dataset_name,
                                                                                   group,
                                                                                   chi_squared_test_results,
                                                                                   expected_freq_stats,
                                                                                   p_value_perm,
                                                                                   p_value_r,
                                                                                   p_value_r_perm,
                                                                                   measure_of_association_contingency_results_list)

            test_results_df_list.append(test_results_df)

        test_results_per_group_df = pd.concat(test_results_df_list, 
                                              ignore_index=True)

        return test_results_per_group_df
    
    def _return_chi_squared_test_results(self,
                                         clusters: np.ndarray,
                                         eval_metrics: np.ndarray,
                                         permute_eval_metrics: bool,
                                         rng: np.random.Generator | None) -> TestResultsChiSquared:

        if permute_eval_metrics:
            eval_metrics = eval_metrics.copy()
            rng.shuffle(eval_metrics)

        observed_freq = crosstab(clusters, eval_metrics).count
        n_observations = np.sum(observed_freq)

        res = chi2_contingency(observed_freq,
                               correction=False)

        chi_squared_results = TestResultsChiSquared(observed_freq,
                                                    res.expected_freq,
                                                    n_observations,
                                                    res.statistic,
                                                    res.dof,
                                                    res.pvalue)


        return chi_squared_results
    
    def _return_chi_squared_perm_p_value(self,
                                         clusters: np.ndarray,
                                         eval_metrics: np.ndarray,
                                         chi_squared_test_results: TestResultsChiSquared) -> float:

        # get chi squared test statistics for permuted data
        rng = np.random.default_rng()
        chi_squared_perm_list = [
            self._return_chi_squared_test_results(clusters, eval_metrics, True, rng).chi_squared_statistic 
            for _ in range(OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES)
        ]

        p_value_perm = np.mean(np.array(chi_squared_perm_list) >= chi_squared_test_results.chi_squared_statistic)

        return p_value_perm

    def _return_expected_frequencies_stats(self,
                                           expected_frequencies: np.ndarray) -> ContingencyExpectedFrequenciesStats:

        if self.evaluation_metric_field_is_categorical:

            has_small_expected_freq_matrix = (expected_frequencies < OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE)
            has_small_expected_freq = has_small_expected_freq_matrix.any()
            n_elements_contingency_table = expected_frequencies.size
            n_elements_contingency_table_below_threshold = has_small_expected_freq_matrix.sum()
            pct_elements_contingency_table_below_threshold = n_elements_contingency_table_below_threshold / n_elements_contingency_table * 100
            
            expected_frequencies_stats = ContingencyExpectedFrequenciesStats(OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE,
                                                                             has_small_expected_freq,
                                                                             n_elements_contingency_table,
                                                                             n_elements_contingency_table_below_threshold,
                                                                             pct_elements_contingency_table_below_threshold)

            return expected_frequencies_stats

        raise ValueError(OMNIBUS_TESTS_ERROR_EVAL_METRIC_NOT_CATEGORICAL_NAME_STR)
    
    def _return_chi_squared_p_value_r(self,
                                      clusters: np.ndarray,
                                      eval_metrics: np.ndarray,
                                      do_permutation_test: bool) -> float:

        with localconverter(ro.default_converter + numpy2ri.converter):
            clusters_r = ro.conversion.py2rpy(clusters)
            eval_metrics_r = ro.conversion.py2rpy(eval_metrics)
        

        clusters_r = r['as.factor'](clusters_r) 
        eval_metrics_r = r['as.factor'](eval_metrics_r) 

        chisq_test_result = r['chisq.test'](clusters_r, 
                                            eval_metrics_r, 
                                            correct=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE, 
                                            simulate_p_value=do_permutation_test, 
                                            B=OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES)
        p_value = chisq_test_result.rx2('p.value')[0]

        return p_value
    
    def _return_measure_association_contingency(self,
                                                observed_freq: np.ndarray,
                                                method: ContingencyEffectSizeEnum) -> float:

        measure_of_association = association(observed_freq,
                                             method=method.name,
                                             correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
        return measure_of_association

    def _return_measure_association_contingency_conf_interval_bootstrap(self,
                                                                        clusters: np.ndarray,
                                                                        eval_metrics: np.ndarray,
                                                                        group: int,
                                                                        method: ContingencyEffectSizeEnum) -> Any:

        def return_measure_association_bootstrap(x: np.ndarray, 
                                                 y: np.ndarray) -> float:
            crosstab_res = crosstab(x, y)
            observed_freq = crosstab_res.count

            # ignore runtime warnings raised when the measure of association could not be calculated due to
            # missing levels of a categorical variable in the contingency table
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                measure_of_association = self._return_measure_association_contingency(observed_freq,
                                                                                      method)

            #TODO: maybe find a better way to exclude measures of association when the measure could not be calculated
            if np.isnan(measure_of_association):
                self._measure_association_fail_dict_per_group[group][method.value] += 1
                return np.random.uniform(low=0.0, high=1.0, size=None)
            else:
                self._measure_association_fail_dict_per_group[group][method.value]
                return measure_of_association

        bootstrap_result = bootstrap((clusters, eval_metrics),
                                      return_measure_association_bootstrap,
                                      n_resamples=OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES,
                                      vectorized=OMNIBUS_TESTS_BOOTSTRAPPING_VECTORIZED,
                                      paired=OMNIBUS_TESTS_BOOTSTRAPPING_PAIRED,
                                      confidence_level=OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                      alternative=OMNIBUS_TESTS_BOOTSTRAPPING_ALTERNATIVE,
                                      method=OMNIBUS_TESTS_BOOTSTRAPPING_METHOD,
                                      random_state=RNG_SEED)

        return bootstrap_result
                            
    def _return_measure_association_contingency_results(self,
                                                        clusters: np.ndarray,
                                                        eval_metrics: np.ndarray,
                                                        observed_freq: np.ndarray,
                                                        group: int) -> list[MeasureAssociationContingencyResults]:

        measure_of_association_contingency_results_list = []
        for method in OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_LIST:

            measure_of_association_type = method.value
            measure_of_association_value = self._return_measure_association_contingency(observed_freq,
                                                                                        method)

            bootstrap_result = self._return_measure_association_contingency_conf_interval_bootstrap(clusters,
                                                                                                    eval_metrics,
                                                                                                    group,
                                                                                                    method)

            measure_of_association_contingency_results = MeasureAssociationContingencyResults(measure_of_association_type,
                                                                                              measure_of_association_value,
                                                                                              OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                                                                              bootstrap_result.confidence_interval,
                                                                                              bootstrap_result.bootstrap_distribution,
                                                                                              bootstrap_result.standard_error)
            measure_of_association_contingency_results_list.append(measure_of_association_contingency_results)

        return measure_of_association_contingency_results_list

    def _return_test_result_df_chi_squared_independence(self,
                                                        dataset_name: str,
                                                        group: int,
                                                        chi_squared_test_results: TestResultsChiSquared,
                                                        expected_frequencies_stats: ContingencyExpectedFrequenciesStats,
                                                        p_val_perm: float,
                                                        p_val_r: float | None,
                                                        p_val_r_perm: float | None,
                                                        measure_of_association_contingency_results_list: list[MeasureAssociationContingencyResults]) -> pd.DataFrame:

        test_results_dict = {DATASET_NAME_FIELD_NAME_STR: dataset_name,
                             GROUP_FIELD_NAME_STR: group,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR: chi_squared_test_results.n_observations,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_FIELD_NAME_STR: expected_frequencies_stats.expected_frequencies_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_FIELD_NAME_STR: expected_frequencies_stats.has_expected_frequency_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_NUMBER_ELEMENTS_FIELD_NAME_STR: expected_frequencies_stats.n_elements_contingency_table,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_NUMBER_ELEMENTS_BELOW_THRESHOLD_FREQ_FIELD_NAME_STR: expected_frequencies_stats.n_elements_contingency_table_expected_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_PCT_ELEMENTS_BELOW_THRESHOLD_FREQ_FIELD_NAME_STR: expected_frequencies_stats.pct_elements_contingency_table_expected_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_TEST_STATISTIC_FIELD_NAME_STR: chi_squared_test_results.chi_squared_statistic,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_DEGREES_OF_FREEDOM_FIELD_NAME_STR: chi_squared_test_results.degrees_of_freedom,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_PVAL_FIELD_NAME_STR: chi_squared_test_results.p_value,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_PERM_PVAL_FIELD_NAME_STR: p_val_perm,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_R_PVAL_FIELD_NAME_STR: p_val_r,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_R_PERM_PVAL_FIELD_NAME_STR: p_val_r_perm,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_PERM_N_PERMS_FIELD_NAME_STR: OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES}

        measure_of_association_result_dict = self._return_measure_of_association_result_dict(measure_of_association_contingency_results_list)

        test_results_dict = test_results_dict | measure_of_association_result_dict

        test_results_df = pd.DataFrame(test_results_dict,
                                       index=(0,))
        return test_results_df

    def _return_omnibus_test_result_aov(self) -> pd.DataFrame:

        test_results_per_group_df = pd.DataFrame()

        test_results_df_list = []
        for group, df in self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR):


            # get anova test results
            anova_test_results = self._return_aov_test_results(df)

            # perform an anova permutation test
            p_value_perm = self._return_aov_perm_p_value(df)

            # perform the anova test in R as sanity check
            if self.include_r_test_results:
                p_value_r, p_value_r_perm = self._return_aov_p_value_r(df)
            else:
                p_value_r, p_value_r_perm = None, None

            # calculate measure of association results
            measure_of_association_aov_results_list = self._return_measure_association_aov_results(df,
                                                                                                   group)
            self._measure_of_association_results_per_group[group] = measure_of_association_aov_results_list

            test_results_df = self._return_test_result_df_aov(self.dataset_name,
                                                              group,
                                                              anova_test_results,
                                                              p_value_perm,
                                                              p_value_r,
                                                              p_value_r_perm,
                                                              measure_of_association_aov_results_list)

            test_results_df_list.append(test_results_df)

        test_results_per_group_df = pd.concat(test_results_df_list, 
                                              ignore_index=True)

        return test_results_per_group_df

    def _return_aov_test_results(self,
                                 sequence_cluster_df: pd.DataFrame) -> TestResultsAOV:

        res = pg.anova(sequence_cluster_df, 
                       dv=self.evaluation_metric_field, 
                       between=CLUSTER_FIELD_NAME_STR, 
                       detailed=True)
        
        n_observations = sequence_cluster_df.shape[0]

        chi_squared_results = TestResultsAOV(n_observations,
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_F_VALUE_FIELD_NAME_STR][0],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_DF_VALUE_FIELD_NAME_STR][0],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_DF_VALUE_FIELD_NAME_STR][1],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_SS_VALUE_FIELD_NAME_STR][0],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_SS_VALUE_FIELD_NAME_STR][1],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_MS_VALUE_FIELD_NAME_STR][0],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_MS_VALUE_FIELD_NAME_STR][1],
                                             res[OMNIBUS_TESTS_CONTINUOUS_AOV_PG_P_VALUE_FIELD_NAME_STR][0])


        return chi_squared_results

    def _calculate_group_mean_variance(self,
                                       sequence_cluster_df: pd.DataFrame,
                                       permute_values: bool,
                                       rng: np.random.Generator | None) -> float:

        if permute_values:
            sequence_cluster_df = sequence_cluster_df.copy()
            rng.shuffle(sequence_cluster_df[self.evaluation_metric_field].values)

        grand_average = sequence_cluster_df[self.evaluation_metric_field].mean()
        n_groups = sequence_cluster_df[CLUSTER_FIELD_NAME_STR].nunique()
        group_mean_variance = (sum((sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                                       .agg(np.mean).values - grand_average) ** 2) / (n_groups - 1))

        return group_mean_variance

    def _return_aov_perm_p_value(self,
                                 sequence_cluster_df: pd.DataFrame) -> float:
        
        group_mean_variance = self._calculate_group_mean_variance(sequence_cluster_df,
                                                                  False,
                                                                  None)

        # get group mean variances for permuted data
        rng = np.random.default_rng()
        group_mean_variance_perm_list = [
            self._calculate_group_mean_variance(sequence_cluster_df, True, rng)
            for _ in range(OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES)
        ]

        p_value_perm = np.mean(np.array(group_mean_variance_perm_list) >= group_mean_variance)

        return p_value_perm

    def _return_aov_p_value_r(self,
                              sequence_cluster_df: pd.DataFrame) -> tuple[float, float]:

        with localconverter(ro.default_converter + pandas2ri.converter):
            sequence_cluster_df_r = ro.conversion.py2rpy(sequence_cluster_df)

        res = r['aovperm'](ro.Formula(f'{self.evaluation_metric_field} ~ C({CLUSTER_FIELD_NAME_STR})'),
                                      data=sequence_cluster_df_r,
                                      np=OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES)

        p_value = res.rx2('table').rx2('parametric P(>F)')[0]
        p_value_perm = res.rx2('table').rx2('resampled P(>F)')[0]

        return p_value, p_value_perm

    def _return_measure_association_aov(self,
                                        sequence_cluster_df: pd.DataFrame,
                                        method: AOVEffectSizeEnum) -> float:

        match method:
            case AOVEffectSizeEnum.ETA_SQUARED:
                measure_of_association = self._return_eta_squared(sequence_cluster_df)
            case AOVEffectSizeEnum.COHENS_F:
                measure_of_association = self._return_eta_squared(sequence_cluster_df)
            case _:
                #TODO: find fitting error
                print('error')

        return measure_of_association

    def _return_eta_squared(self,
                            sequence_cluster_df: pd.DataFrame) -> float:

        grand_mean = sequence_cluster_df[self.evaluation_metric_field].mean()
        ss_between = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                         .transform(lambda x: (np.mean(x) - grand_mean)**2).sum())
        ss_within = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                        .transform(lambda x: (x - np.mean(x))**2).sum())

        eta_squared = ss_between / (ss_within + ss_between)
    
        return eta_squared

    def _return_cohens_f(self,
                         sequence_cluster_df: pd.DataFrame) -> float:

        grand_mean = sequence_cluster_df[self.evaluation_metric_field].mean()
        ss_between = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                         .transform(lambda x: (np.mean(x) - grand_mean)**2).sum())
        ss_within = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                        .transform(lambda x: (x - np.mean(x))**2).sum())

        eta_squared = ss_between / (ss_within + ss_between)

        cohens_f = (eta_squared / (1 - eta_squared)) ** (1/2)
    
        return cohens_f

    def _return_measure_association_aov_conf_interval_bootstrap(self,
                                                                sequence_cluster_df: pd.DataFrame,
                                                                group: int,
                                                                method: AOVEffectSizeEnum) -> Any:

        def return_measure_association_bootstrap(x: np.ndarray, 
                                                 y: np.ndarray) -> float:

            sequence_cluster_df = pd.DataFrame({CLUSTER_FIELD_NAME_STR: x,
                                                self.evaluation_metric_field: y})

            measure_of_association = self._return_measure_association_aov(sequence_cluster_df,
                                                                          method)

            #TODO: maybe find a better way to exclude measures of association when the measure could not be calculated
            if np.isnan(measure_of_association):
                self._measure_association_fail_dict_per_group[group][method.value] += 1
                return np.random.uniform(low=0.0, high=1.0, size=None)
            else:
                self._measure_association_fail_dict_per_group[group][method.value]
                return measure_of_association

        clusters = sequence_cluster_df[CLUSTER_FIELD_NAME_STR].values
        eval_metrics = sequence_cluster_df[self.evaluation_metric_field].values

        bootstrap_result = bootstrap((clusters, eval_metrics),
                                      return_measure_association_bootstrap,
                                      n_resamples=OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES,
                                      vectorized=OMNIBUS_TESTS_BOOTSTRAPPING_VECTORIZED,
                                      paired=OMNIBUS_TESTS_BOOTSTRAPPING_PAIRED,
                                      confidence_level=OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                      alternative=OMNIBUS_TESTS_BOOTSTRAPPING_ALTERNATIVE,
                                      method=OMNIBUS_TESTS_BOOTSTRAPPING_METHOD,
                                      random_state=RNG_SEED)

        return bootstrap_result

    def _return_measure_association_aov_results(self,
                                                sequence_cluster_df: pd.DataFrame,
                                                group: int) -> list[MeasureAssociationAOVResults]:

        measure_of_association_aov_results_list = []
        for method in OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_LIST:

            measure_of_association_type = method.value
            measure_of_association_value = self._return_measure_association_aov(sequence_cluster_df,
                                                                                method)

            bootstrap_result = self._return_measure_association_aov_conf_interval_bootstrap(sequence_cluster_df,
                                                                                            group,
                                                                                            method)

            measure_of_association_aov_results = MeasureAssociationAOVResults(measure_of_association_type,
                                                                              measure_of_association_value,
                                                                              OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                                                              bootstrap_result.confidence_interval,
                                                                              bootstrap_result.bootstrap_distribution,
                                                                              bootstrap_result.standard_error)

            measure_of_association_aov_results_list.append(measure_of_association_aov_results)

        return measure_of_association_aov_results_list

    def _return_test_result_df_aov(self,
                                   dataset_name: str,
                                   group: int,
                                   anova_test_results: TestResultsAOV,
                                   p_val_perm: float,
                                   p_val_r: float | None,
                                   p_val_r_perm: float | None,
                                   measure_of_association_aov_results_list: list[MeasureAssociationAOVResults]) -> pd.DataFrame:

        test_results_dict = {DATASET_NAME_FIELD_NAME_STR: dataset_name,
                             GROUP_FIELD_NAME_STR: group,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR: anova_test_results.n_observations,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_F_TEST_STATISTIC_FIELD_NAME_STR: anova_test_results.f_statistic,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_DOF_BETWEEN_FIELD_NAME_STR: anova_test_results.degrees_of_freedom_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_DOF_WITHIN_FIELD_NAME_STR: anova_test_results.degrees_of_freedom_within,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_SS_BETWEEN_FIELD_NAME_STR: anova_test_results.ss_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_SS_WITHIN_FIELD_NAME_STR: anova_test_results.ss_within,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_MSS_BETWEEN_FIELD_NAME_STR: anova_test_results.mss_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_MSS_WITHIN_FIELD_NAME_STR: anova_test_results.mss_within,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_PVAL_FIELD_NAME_STR: anova_test_results.p_value,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_PERM_PVAL_FIELD_NAME_STR: p_val_perm,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_R_PVAL_FIELD_NAME_STR: p_val_r,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_R_PERM_PVAL_FIELD_NAME_STR: p_val_r_perm,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_PERM_N_PERMS_FIELD_NAME_STR: OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES}

        measure_of_association_result_dict = self._return_measure_of_association_result_dict(measure_of_association_aov_results_list)

        test_results_dict = test_results_dict | measure_of_association_result_dict

        test_results_df = pd.DataFrame(test_results_dict,
                                       index = (0,))
        return test_results_df

    def _return_measure_of_association_result_dict(self,
                                                   measure_of_association_results_list: list[MeasureAssociationContingencyResults | MeasureAssociationAOVResults]) -> dict:

        measure_of_association_result_dict = {}
        for result in measure_of_association_results_list:

            measure_of_association_value_field_name = result.measure_of_association_type + OMNIBUS_TESTS_EFFECT_SIZE_VALUE_FIELD_NAME_STR
            measure_of_association_conf_int_value_field_name = result.measure_of_association_type + OMNIBUS_TESTS_EFFECT_SIZE_CONF_INT_VALUE_FIELD_NAME_STR

            measure_of_association_result = {measure_of_association_value_field_name: result.measure_of_association_value,
                                             measure_of_association_conf_int_value_field_name: [tuple(result.measure_of_association_conf_int)]}
            
            measure_of_association_result_dict = measure_of_association_result_dict | measure_of_association_result

        additional_info_dict = {OMNIBUS_TESTS_EFFECT_SIZE_CONF_INT_ALPHA_FIELD_NAME_STR: result.measure_of_association_conf_int_level,
                                OMNIBUS_TESTS_EFFECT_SIZE_CONF_INT_N_BOOTSTRAP_SAMPLES_FIELD_NAME_STR: OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES}

        measure_of_association_result_dict = measure_of_association_result_dict | additional_info_dict

        return measure_of_association_result_dict
    
    def _return_seq_cluster_eval_metric_per_group_df(self,
                                                     sequence_cluster_per_group_df: pd.DataFrame,
                                                     interactions: pd.DataFrame) -> pd.DataFrame:

        #TODO: maybe incorporate different measures for the evaluation metric here(eg. highest score....)
        # now it only works with group wide aggregates!
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