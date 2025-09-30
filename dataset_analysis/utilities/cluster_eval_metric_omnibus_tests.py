from .configs.general_config import *
from .configs.omnibus_tests_config import *
from .constants.constants import *
from .constants.enums import *
from .standard_import import *
from .plotting_functions import *
from .validators import *
from .data_classes import *

class ClusterEvalMetricOmnibusTest():
    """docstring for ClassName."""

    def __init__(self, 
                 dataset_name: str,
                 interactions: pd.DataFrame,
                 sequence_cluster_per_group_df: pd.DataFrame,
                 evaluation_metric_field: str,
                 evaluation_metric_is_categorical: bool,
                 exclude_non_clustered: bool,
                 include_r_test_results: bool,
                 parallelize_computation: bool) -> None:

        self.dataset_name: str = dataset_name
        self.interactions: pd.DataFrame = copy.deepcopy(interactions)
        self.sequence_cluster_per_group_df: pd.DataFrame = sequence_cluster_per_group_df
        self.evaluation_metric_field: str = evaluation_metric_field
        self.evaluation_metric_field_is_categorical: bool = evaluation_metric_is_categorical
        self.exclude_non_clustered: bool = exclude_non_clustered
        self.include_r_test_results: bool = include_r_test_results
        self.parallelize_computation = parallelize_computation
        self._omnibus_tests_results: List[OmnibusTestResults] = []

        # initialization of p-value fields list
        self._p_val_field_name_list = [OMNIBUS_TESTS_PVAL_FIELD_NAME_STR,
                                       OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR,
                                       OMNIBUS_TESTS_R_PVAL_FIELD_NAME_STR,
                                       OMNIBUS_TESTS_R_PERM_PVAL_FIELD_NAME_STR]

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
        
        # correct p-values for multiple testing
        self._perform_p_value_correction(self.omnibus_test_result_df)

        # add p-value is significant fields for alpha specified in config
        self._add_p_value_is_significant_fields(self.omnibus_test_result_df)

        # round results
        self.omnibus_test_result_df = self.omnibus_test_result_df.round(OMNIBUS_TEST_RESULTS_ROUND_N_DIGITS)

        # add evaluation metric information
        self.omnibus_test_result_df.insert(2,
                                           OMNIBUS_TESTS_EVAlUATION_FIELD_TYPE_FIELD_NAME_STR,
                                           self.evaluation_metric_field)
        self.omnibus_test_result_df.insert(3,
                                           OMNIBUS_TESTS_EVAlUATION_FIELD_IS_CATEGORICAL_FIELD_NAME_STR,
                                           self.evaluation_metric_field_is_categorical)
    
    def return_omnibus_test_result(self) -> pd.DataFrame:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)
        
        return self.omnibus_test_result_df

    def return_measure_association_conf_int_bootstrap_failures(self) -> pd.DataFrame:

        _ = check_if_not_empty(self._omnibus_tests_results,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        moa_fail_dict_list = []
        for res in self._omnibus_tests_results:
            moa_fail_dict = {GROUP_FIELD_NAME_STR: res.group}
            for fail_dict in res.measure_of_association_fail_dict:
                for moa, fail_count in fail_dict.items():

                    fail_count_pct = (fail_count / OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES * 100)

                    moa_fail_count_field_name = moa + OMNIBUS_TESTS_MEASURE_ASSOCIATION_BOOTSTRAP_CONF_INT_FAIL_COUNT_FIELD_NAME_STR
                    moa_fail_pct_field_name = moa + OMNIBUS_TESTS_MEASURE_ASSOCIATION_BOOTSTRAP_CONF_INT_FAIL_PCT_FIELD_NAME_STR

                    moa_fail_dict[moa_fail_count_field_name] = fail_count
                    moa_fail_dict[moa_fail_pct_field_name] = fail_count_pct
            moa_fail_dict_list.append(moa_fail_dict)

        measure_association_conf_int_fail_count = pd.DataFrame(moa_fail_dict_list)
        measure_association_conf_int_fail_count = measure_association_conf_int_fail_count.sort_values(by=GROUP_FIELD_NAME_STR, ascending=True)

        return measure_association_conf_int_fail_count 
    
    def return_measure_association_results_per_group_dict(self) -> dict[int, list[MeasureAssociationContingencyResults] | list[MeasureAssociationAOVResults]]:

        _ = check_if_not_empty(self._omnibus_tests_results,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        measure_of_association_results_per_group = {res.group: res.measure_of_association_results for res in self._omnibus_tests_results}

        return measure_of_association_results_per_group

    def plot_cluster_eval_metric_per_group_plot(self) -> None:

        _ = check_if_not_empty(self.omnibus_test_result_df,
                               OMNIBUS_TESTS_ERROR_NO_TEST_RESULTS_NAME_STR)

        if self.evaluation_metric_field_is_categorical:
            self._plot_cluster_eval_metric_mosaic_plot()
        else:
            self._plot_cluster_eval_metric_boxplot()
    
    def add_omnibus_test_result_to_results_tables(self,
                                                  result_tables: Type[Any]) -> None:

        # add data to results_table
        result_tables.omnibus_test_result_df = self.return_omnibus_test_result().copy()
        result_tables.measure_association_conf_int_bootstrap_failures_df = self.return_measure_association_conf_int_bootstrap_failures().copy()
    
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
        
        self._add_plot_data_title(g,
                                  OMNIBUS_TESTS_P_VALUE_CORRECTION_PLOT_INCLUDE,
                                  OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE,
                                  OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_PLOT_INCLUDE,
                                  FACET_GRID_SUBPLOTS_H_SPACE_SQUARE_WITH_TITLE)

        self._adjust_axis_labels(g)

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

        self._add_plot_data_title(g,
                                  OMNIBUS_TESTS_P_VALUE_CORRECTION_PLOT_INCLUDE,
                                  OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_PLOT_INCLUDE,
                                  OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_PLOT_INCLUDE,
                                  FACET_GRID_SUBPLOTS_H_SPACE_SQUARE_WITH_TITLE)
        
        self._adjust_axis_labels(g)

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

    def _adjust_axis_labels(self,
                            g: FacetGrid) -> None:
        
        for ax in g.axes.flat:

            if OMNIBUS_TESTS_ADJUST_X_LABEL:

                y_label = ax.get_xlabel()
            
                y_label = self._transform_label(y_label,
                                                OMNIBUS_TESTS_X_LABEL_SPLIT_STRING,
                                                OMNIBUS_TESTS_X_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT)

                ax.set_xlabel(y_label, 
                              va='top',
                              ma='center',
                              rotation=OMNIBUS_TESTS_X_LABEL_ROTATION, 
                              labelpad=OMNIBUS_TESTS_X_LABEL_VERTICAL_PADDING)  

            if OMNIBUS_TESTS_ADJUST_Y_LABEL:

                y_label = ax.get_ylabel()
            
                y_label = self._transform_label(y_label,
                                                OMNIBUS_TESTS_Y_LABEL_SPLIT_STRING,
                                                OMNIBUS_TESTS_Y_LABEL_WORDS_PER_LINE_THRESHOLD_COUNT)

                ax.set_ylabel(y_label, 
                              ha='right',
                              ma='center',
                              rotation=OMNIBUS_TESTS_Y_LABEL_ROTATION, 
                              labelpad=OMNIBUS_TESTS_Y_LABEL_RIGHT_PADDING)  

    def _transform_label(self,
                         label: str,
                         split_by_str: str,
                         words_per_line: int) -> str:

        n_words_label = len(label.split(split_by_str))

        if n_words_label > words_per_line:

            label_words = label.split(split_by_str)

            label = ''
            for n, word in enumerate(label_words):

                if n == 0:
                    label += word
                elif n % words_per_line == 0:
                    word = '\n' + word
                    label += word
                else:
                    word = split_by_str + word
                    label += word
        
        return label
            
    def _add_plot_data_title(self,
                             sns_plot,
                             p_value_correction_method: PValueCorrectionEnum,
                             measure_of_association_method: ContingencyMeasureAssociationEnum | AOVMeasueAssociationEnum,
                             measure_of_association_strength_guideline: ContingencyMeasureAssociationStrengthGuidelineEnum | AOVMeasureAssociationStrengthGuidelineEnum,
                             h_space_title: int | float) -> None:

        axes_iterable = zip(sns_plot.axes.flat, sns_plot.facet_data())
        for ax, (_, subset) in axes_iterable:

            p_value_correction_value_field_name = OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + p_value_correction_method.value 
            measure_of_association_value_field_name = measure_of_association_method.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR
            measure_of_association_conf_int_value_field_name = measure_of_association_method.value + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR
            measure_of_association_strength_guideline_field_name = (measure_of_association_method.value +
                                                                    '_' + 
                                                                    measure_of_association_strength_guideline.value + 
                                                                    OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR)

            measure_of_association_strength_guideline_ci_lower_field_name = (measure_of_association_strength_guideline_field_name +
                                                                             OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_LOWER_FIELD_NAME_STR)
            measure_of_association_strength_guideline_ci_upper_field_name = (measure_of_association_strength_guideline_field_name +
                                                                             OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_UPPER_FIELD_NAME_STR)

            if self.evaluation_metric_field_is_categorical:
                has_exp_freq_below_t_field = OMNIBUS_TESTS_CONTINGENCY_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_FIELD_NAME_STR
                test_statistic_field = OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_TEST_STATISTIC_FIELD_NAME_STR
            else:
                test_statistic_field = OMNIBUS_TESTS_CONTINUOUS_AOV_F_TEST_STATISTIC_FIELD_NAME_STR

            # extract data for plotting
            group = subset[GROUP_FIELD_NAME_STR].iloc[0]
            is_group_series = self.omnibus_test_result_df[GROUP_FIELD_NAME_STR] == group
            n_observations = self.omnibus_test_result_df.loc[is_group_series, OMNIBUS_TESTS_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR].values[0] 
            if self.evaluation_metric_field_is_categorical:
                has_expected_frequency_below_threshold = self.omnibus_test_result_df.loc[is_group_series, has_exp_freq_below_t_field].values[0] 
            test_statistic = round(self.omnibus_test_result_df.loc[is_group_series, test_statistic_field].values[0], 3)
            p_value_perm = round(self.omnibus_test_result_df.loc[is_group_series, OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR].values[0], 3)
            p_value_perm_corrected = round(self.omnibus_test_result_df.loc[is_group_series, p_value_correction_value_field_name].values[0], 3)
            measure_of_association_type = measure_of_association_value_field_name
            measure_of_association_value = round(self.omnibus_test_result_df.loc[is_group_series, measure_of_association_value_field_name].values[0], 3)
            measure_of_association_conf_int = self.omnibus_test_result_df.loc[is_group_series, measure_of_association_conf_int_value_field_name].values[0]
            measure_of_association_conf_int = tuple(map(lambda x: round(x, 3), measure_of_association_conf_int))
            measure_of_association_conf_int_lvl = int(OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL * 100)
            measure_of_association_strength_guideline_value = self.omnibus_test_result_df.loc[is_group_series, measure_of_association_strength_guideline_field_name].values[0]
            measure_of_association_strength_guideline_ci_lower_value = self.omnibus_test_result_df.loc[is_group_series, measure_of_association_strength_guideline_ci_lower_field_name].values[0]
            measure_of_association_strength_guideline_ci_upper_value = self.omnibus_test_result_df.loc[is_group_series, measure_of_association_strength_guideline_ci_upper_field_name].values[0]

    
            # generate title strings for plot
            p_value_perm_star_str = self._return_p_value_star_string(p_value_perm)
            p_value_perm_corrected_star_str = self._return_p_value_star_string(p_value_perm_corrected)

            group_str = f'{GROUP_FIELD_NAME_STR}: {group}'

            n_observations_str = f'\n{OMNIBUS_TESTS_NUMBER_OBSERVATIONS_PLOT_VALUE_NAME_STR}: {n_observations}'

            if self.evaluation_metric_field_is_categorical:
                has_expected_frequency_below_threshold_str = f'\n{OMNIBUS_TESTS_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_PLOT_VALUE_NAME_STR}{OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE}: {has_expected_frequency_below_threshold}'
            else:
                has_expected_frequency_below_threshold_str = ''

            if self.evaluation_metric_field_is_categorical:
                test_statistic_str = f'\n{OMNIBUS_TESTS_CHI_SQUARED_PLOT_VALUE_NAME_STR}: {test_statistic}'
            else:
                test_statistic_str = f'\n{OMNIBUS_TESTS_F_PLOT_VALUE_NAME_STR}: {test_statistic}'

            p_value_perm_str = f'\n{OMNIBUS_TESTS_PVAL_PERM_PLOT_VALUE_NAME_STR}: ' + p_value_perm_star_str
            p_value_perm_corrected_str = f'\n{OMNIBUS_TESTS_PVAL_PERM_CORRECTED_PLOT_VALUE_NAME_STR}: ' + p_value_perm_corrected_star_str
            p_value_correction_method_str = f'\n{OMNIBUS_TESTS_PVAL_CORRECTION_METHOD_PLOT_VALUE_NAME_STR}: ' + p_value_correction_method.value

            sub_strings = measure_of_association_type.split('_')
            measure_of_association_type = '_'.join([sub_str.capitalize() for sub_str in sub_strings])
            measure_of_association_str = f'\n{measure_of_association_type}: {measure_of_association_value}'
            measure_of_association_conf_int_str = f'\n{measure_of_association_conf_int_lvl}% {OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_PLOT_VALUE_NAME_STR}: {measure_of_association_conf_int}'

            sub_strings = measure_of_association_strength_guideline.value.split('_')
            measure_of_association_strength_guideline_type = '_'.join([sub_str.capitalize() for sub_str in sub_strings])
            measure_of_association_strength_guideline_type = OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_PLOT_VALUE_NAME_STR + measure_of_association_strength_guideline_type
            measure_of_association_strength_guideline_str = (f'\n{measure_of_association_strength_guideline_type}:' +
                                                             f'\n{measure_of_association_strength_guideline_ci_lower_value} - ' +
                                                             f'{measure_of_association_strength_guideline_value}' + 
                                                             f' - {measure_of_association_strength_guideline_ci_upper_value}')

            title_str = ''.join((group_str,
                                 n_observations_str,
                                 has_expected_frequency_below_threshold_str,
                                 test_statistic_str,
                                 p_value_perm_str,
                                 p_value_perm_corrected_str,
                                 p_value_correction_method_str,
                                 measure_of_association_str,
                                 measure_of_association_conf_int_str,
                                 measure_of_association_strength_guideline_str))
            ax.set_title(title_str)
        
        plt.subplots_adjust(hspace=h_space_title)
    
    def _return_p_value_star_string(self,
                                    p_value: float) -> str:
        if p_value > OMNIBUS_TESTS_ALPHA_LEVEL:
            star_str = ''
        elif OMNIBUS_TESTS_TWO_STAR_UPPER_BOUND < p_value <= OMNIBUS_TESTS_ONE_STAR_UPPER_BOUND:
            star_str = '*'
        elif OMNIBUS_TESTS_THREE_STAR_UPPER_BOUND < p_value <= OMNIBUS_TESTS_TWO_STAR_UPPER_BOUND:
            star_str = '**'
        else:
            star_str = '***'

        if p_value <= OMNIBUS_TESTS_ALPHA_LEVEL:
            p_value_star_str = f'$\mathbf{ {p_value} }$ $\mathbf{ {star_str} }$'
        else:
            p_value_star_str = f'{p_value}{star_str}'
        
        return p_value_star_str

    def _return_omnibus_test_result_categorical_var(self) -> pd.DataFrame:

        omnibus_test_results = self._return_omnibus_test_result(self._return_omnibus_test_result_chi_squared_independence)

        return omnibus_test_results

    def _return_omnibus_test_result_continuous_var(self) -> pd.DataFrame:

        omnibus_test_results = self._return_omnibus_test_result(self._return_omnibus_test_result_aov)

        return omnibus_test_results
    
    def _return_omnibus_test_result(self,
                                    omnibus_test_function: Callable) -> pd.DataFrame:

        if self.parallelize_computation:
            self._omnibus_tests_results = (Parallel(n_jobs=NUMBER_OF_CORES)
                                           (delayed(omnibus_test_function)
                                            (group,
                                             df) for group, df in tqdm(self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR))))
        else:
            self._omnibus_tests_results = [omnibus_test_function(group, df) for group, df in tqdm(self.sequence_cluster_eval_metric_per_group_df.groupby(GROUP_FIELD_NAME_STR))]

        test_results_df_list = [res.test_result_df for res in self._omnibus_tests_results]

        test_results_per_group_df = pd.concat(test_results_df_list, 
                                              ignore_index=True)
        
        test_results_per_group_df = test_results_per_group_df.sort_values(by=GROUP_FIELD_NAME_STR, ascending=True)

        return test_results_per_group_df

    def _return_omnibus_test_result_chi_squared_independence(self,
                                                             group: int,
                                                             sequence_cluster_eval_metric_df: pd.DataFrame) -> OmnibusTestResults:

        clusters = sequence_cluster_eval_metric_df[CLUSTER_FIELD_NAME_STR].values
        eval_metrics = sequence_cluster_eval_metric_df[self.evaluation_metric_field].values

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
        measure_of_association_contingency_results_list, measure_of_association_contingency_fail_dict_list = self._return_measure_association_contingency_results(clusters,
                                                                                                                                                                  eval_metrics,
                                                                                                                                                                  chi_squared_test_results.observed_frequency)

        test_results_df = self._return_test_result_df_chi_squared_independence(self.dataset_name,
                                                                               group,
                                                                               chi_squared_test_results,
                                                                               expected_freq_stats,
                                                                               p_value_perm,
                                                                               p_value_r,
                                                                               p_value_r_perm,
                                                                               measure_of_association_contingency_results_list)



        return OmnibusTestResults(group,
                                  test_results_df,
                                  measure_of_association_contingency_results_list,
                                  measure_of_association_contingency_fail_dict_list)
    
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
            dimensions_contingency_table = expected_frequencies.shape
            
            expected_frequencies_stats = ContingencyExpectedFrequenciesStats(OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_VALUE,
                                                                             has_small_expected_freq,
                                                                             n_elements_contingency_table,
                                                                             n_elements_contingency_table_below_threshold,
                                                                             pct_elements_contingency_table_below_threshold,
                                                                             dimensions_contingency_table)

            return expected_frequencies_stats

        raise ValueError(OMNIBUS_TESTS_ERROR_EVAL_METRIC_NOT_CATEGORICAL_NAME_STR)
    
    def _return_chi_squared_p_value_r(self,
                                      clusters: np.ndarray,
                                      eval_metrics: np.ndarray,
                                      do_permutation_test: bool) -> float:

        rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None 
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None

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
                                                method: ContingencyMeasureAssociationEnum) -> float:

        match method:
            case ContingencyMeasureAssociationEnum.CRAMER:
                measure_of_association = self._return_cramers_v(observed_freq)
            case ContingencyMeasureAssociationEnum.CRAMER_BIAS_CORRECTED:
                measure_of_association = self._return_cramers_v_bias_corrected(observed_freq)
            case ContingencyMeasureAssociationEnum.TSCHUPROW:
                measure_of_association = self._return_tschuprows_t(observed_freq)
            case ContingencyMeasureAssociationEnum.PEARSON:
                measure_of_association = self._return_pearsons_c(observed_freq)
            case _:
                raise ValueError(OMNIBUS_TESTS_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{method}')

        return measure_of_association
    
    def _return_cramers_v(self,
                          observed_freq: np.ndarray) -> float:
        """
        ---
        [1] H. Cramer, MATHEMATICAL METHODS OF STATISTICS. in Princeton landmarks in mathematics and physics. Princeton University Press, 1946.
        [2] D. J. Sheskin, Handbook of parametric and nonparametric statistical procedures, Fifth edition. Boca Raton London New York: CRC Press, Taylor & Francis Group, 2011. doi: 10.1201/9780429186196.
        """

        cramers_v = association(observed_freq,
                                method='cramer',
                                correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
        return cramers_v

    def _return_tschuprows_t(self,
                             observed_freq: np.ndarray) -> float:

        tschuprows_t = association(observed_freq,
                                   method='tschuprow',
                                   correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
        return tschuprows_t

    def _return_pearsons_c(self,
                           observed_freq: np.ndarray) -> float:

        pearsons_c = association(observed_freq,
                                 method='pearson',
                                 correction=OMNIBUS_TESTS_CONTINGENCY_YATES_CORRECTION_VALUE)
        return pearsons_c

    def _return_cramers_v_bias_corrected(self,
                                         observed_freq: np.ndarray) -> float:
        """
        ---
        [1] W. Bergsma, “A bias-correction for Cramér’s and Tschuprow’s,” Journal of the Korean Statistical Society, vol. 42, no. 3, pp. 323–328, Sep. 2013, doi: 10.1016/j.jkss.2012.10.002.
        """


        chi2_stat = sp.stats.chi2_contingency(observed_freq).statistic

        n_observations = np.sum(observed_freq)
        phi_squared = chi2_stat / n_observations
        n_rows = observed_freq.shape[0]
        n_cols = observed_freq.shape[1]

        phi_squared_bias_corrected = phi_squared - ((1 / (n_observations - 1)) * (n_rows - 1) * (n_cols - 1))

        phi_squared_bias_corrected_non_neg = max(0, phi_squared_bias_corrected)

        r_tilde = n_rows - (1 / (n_observations - 1)) * (n_rows - 1) ** 2
        c_tilde = n_cols - (1 / (n_observations - 1)) * (n_cols - 1) ** 2

        cramers_v_squared_bias_corrected = phi_squared_bias_corrected_non_neg / min(r_tilde - 1, c_tilde - 1)
        cramers_v_bias_corrected = math.sqrt(cramers_v_squared_bias_corrected)

        return cramers_v_bias_corrected

    def _return_measure_association_contingency_conf_interval_bootstrap(self,
                                                                        clusters: np.ndarray,
                                                                        eval_metrics: np.ndarray,
                                                                        method: ContingencyMeasureAssociationEnum) -> tuple[Any, DefaultDict[str, int]]:

        measure_association_fail_dict = defaultdict(int)
        measure_association_fail_dict[method.value]
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

            if method == ContingencyMeasureAssociationEnum.CRAMER_BIAS_CORRECTED:
                if measure_of_association == 0:
                    measure_of_association = np.random.uniform(low=0.0, high=0.000001, size=None)

            if np.isnan(measure_of_association):
                measure_association_fail_dict[method.value] += 1
                return 0
            else:
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

        return bootstrap_result, measure_association_fail_dict

    def _return_measure_association_contingency_strength_value(self,
                                                               measure_of_association_value: float,
                                                               strength_guideline_method: ContingencyMeasureAssociationStrengthGuidelineEnum) -> str:

        match strength_guideline_method:
            case ContingencyMeasureAssociationStrengthGuidelineEnum.COHEN_1988:
                measure_of_association_strength = self._return_cohen_1988_measure_association_contingency_strength(measure_of_association_value)
            case ContingencyMeasureAssociationStrengthGuidelineEnum.GIGNAC_SZODORAI_2016:
                measure_of_association_strength = self._return_gignac_szodorai_2016_measure_association_contingency_strength(measure_of_association_value)
            case ContingencyMeasureAssociationStrengthGuidelineEnum.FUNDER_OZER_2019:
                measure_of_association_strength = self._return_funder_ozer_2019_measure_association_contingency_strength(measure_of_association_value)
            case ContingencyMeasureAssociationStrengthGuidelineEnum.LOVAKOV_AGADULLINA_2021:
                measure_of_association_strength = self._return_lovakov_agadullina_2021_measure_association_contingency_strength(measure_of_association_value)
            case _:
                raise ValueError(OMNIBUS_TESTS_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{strength_guideline_method}')

        return measure_of_association_strength

    def _return_cohen_1988_measure_association_contingency_strength(self,
                                                                    measure_of_association_value: float) -> str:
        
        moa_guideline = Cohen1988MeasureAssociationStrengthContingency()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)
        
        return moa_strength.value

    def _return_gignac_szodorai_2016_measure_association_contingency_strength(self,
                                                                              measure_of_association_value: float) -> str:

        moa_guideline = GignacSzodorai2016MeasureAssociationStrengthContingency()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)
        
        return moa_strength.value

    def _return_funder_ozer_2019_measure_association_contingency_strength(self,
                                                                          measure_of_association_value: float) -> str:

        moa_guideline = FunderOzer2019MeasureAssociationStrengthContingency()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)
        
        return moa_strength.value

    def _return_lovakov_agadullina_2021_measure_association_contingency_strength(self,
                                                                                 measure_of_association_value: float) -> str:

        moa_guideline = LovakovAgadullina2021MeasureAssociationStrengthContingency()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)
        
        return moa_strength.value
                            
    def _return_measure_association_contingency_results(self,
                                                        clusters: np.ndarray,
                                                        eval_metrics: np.ndarray,
                                                        observed_freq: np.ndarray) -> tuple[list[MeasureAssociationContingencyResults], list[DefaultDict[str, int]]]:

        measure_of_association_contingency_results_list = []
        measure_of_association_contingency_fail_dict_list = []
        for moa_method, moa_strength_guide_method_list in OMNIBUS_TESTS_CONTINGENCY_MEASURE_OF_ASSOCIATION_DICT.items():

            measure_of_association_type = moa_method.value
            measure_of_association_value = self._return_measure_association_contingency(observed_freq,
                                                                                        moa_method)
            
            moa_strength_guide_methods = [moa_strength_guide_method.value for moa_strength_guide_method in moa_strength_guide_method_list]
            moa_strength_guide_values = [self._return_measure_association_contingency_strength_value(measure_of_association_value, 
                                                                                                     moa_strength_guide_method)
                                            for moa_strength_guide_method in moa_strength_guide_method_list]

            bootstrap_result, measure_association_fail_dict = self._return_measure_association_contingency_conf_interval_bootstrap(clusters,
                                                                                                                                   eval_metrics,
                                                                                                                                   moa_method)

            moa_strength_guide_conf_int_lower_bound_values = [self._return_measure_association_contingency_strength_value(bootstrap_result.confidence_interval[0], 
                                                                                                                          moa_strength_guide_method)
                                                              for moa_strength_guide_method in moa_strength_guide_method_list]

            moa_strength_guide_conf_int_upper_bound_values = [self._return_measure_association_contingency_strength_value(bootstrap_result.confidence_interval[1], 
                                                                                                                          moa_strength_guide_method)
                                                              for moa_strength_guide_method in moa_strength_guide_method_list]

            measure_of_association_contingency_results = MeasureAssociationContingencyResults(measure_of_association_type,
                                                                                              measure_of_association_value,
                                                                                              OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                                                                              bootstrap_result.confidence_interval,
                                                                                              bootstrap_result.bootstrap_distribution,
                                                                                              bootstrap_result.standard_error,
                                                                                              moa_strength_guide_methods,
                                                                                              moa_strength_guide_values,
                                                                                              moa_strength_guide_conf_int_lower_bound_values,
                                                                                              moa_strength_guide_conf_int_upper_bound_values)

            measure_of_association_contingency_results_list.append(measure_of_association_contingency_results)
            measure_of_association_contingency_fail_dict_list.append(measure_association_fail_dict)

        return measure_of_association_contingency_results_list, measure_of_association_contingency_fail_dict_list

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
                             OMNIBUS_TESTS_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR: chi_squared_test_results.n_observations,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_FREQ_THRESHOLD_FIELD_NAME_STR: expected_frequencies_stats.expected_frequencies_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_HAS_EXPECTED_FREQ_BELOW_THRESHOLD_FIELD_NAME_STR: expected_frequencies_stats.has_expected_frequency_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_NUMBER_ELEMENTS_FIELD_NAME_STR: expected_frequencies_stats.n_elements_contingency_table,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_NUMBER_ELEMENTS_BELOW_THRESHOLD_FREQ_FIELD_NAME_STR: expected_frequencies_stats.n_elements_contingency_table_expected_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_EXPECTED_PCT_ELEMENTS_BELOW_THRESHOLD_FREQ_FIELD_NAME_STR: expected_frequencies_stats.pct_elements_contingency_table_expected_below_threshold,
                             OMNIBUS_TESTS_CONTINGENCY_TABLE_DIMENSIONS_FIELD_NAME_STR: [expected_frequencies_stats.table_dimensions],
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_TEST_STATISTIC_FIELD_NAME_STR: chi_squared_test_results.chi_squared_statistic,
                             OMNIBUS_TESTS_CONTINGENCY_CHI_SQUARED_DEGREES_OF_FREEDOM_FIELD_NAME_STR: chi_squared_test_results.degrees_of_freedom,
                             OMNIBUS_TESTS_PVAL_FIELD_NAME_STR: chi_squared_test_results.p_value,
                             OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR: p_val_perm,
                             OMNIBUS_TESTS_R_PVAL_FIELD_NAME_STR: p_val_r,
                             OMNIBUS_TESTS_R_PERM_PVAL_FIELD_NAME_STR: p_val_r_perm,
                             OMNIBUS_TESTS_PERM_N_PERMS_FIELD_NAME_STR: OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES}

        measure_of_association_result_dict = self._return_measure_of_association_result_dict(measure_of_association_contingency_results_list)

        test_results_dict = test_results_dict | measure_of_association_result_dict

        test_results_df = pd.DataFrame(test_results_dict,
                                       index=(0,))
        return test_results_df

    def _return_omnibus_test_result_aov(self,
                                        group: int,
                                        sequence_cluster_eval_metric_df: pd.DataFrame) -> OmnibusTestResults:

        # get anova test results
        anova_test_results = self._return_aov_test_results(sequence_cluster_eval_metric_df)

        # perform an anova permutation test
        p_value_perm = self._return_aov_perm_p_value(sequence_cluster_eval_metric_df)

        # perform the anova test in R as sanity check
        if self.include_r_test_results:
            p_value_r, p_value_r_perm = self._return_aov_p_value_r(sequence_cluster_eval_metric_df)
        else:
            p_value_r, p_value_r_perm = None, None

        # calculate measure of association results
        measure_of_association_aov_results_list, measure_of_association_aov_fail_dict_list = self._return_measure_association_aov_results(sequence_cluster_eval_metric_df)

        test_results_df = self._return_test_result_df_aov(self.dataset_name,
                                                          group,
                                                          anova_test_results,
                                                          p_value_perm,
                                                          p_value_r,
                                                          p_value_r_perm,
                                                          measure_of_association_aov_results_list)
        return OmnibusTestResults(group,
                                  test_results_df,
                                  measure_of_association_aov_results_list,
                                  measure_of_association_aov_fail_dict_list)

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

        rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None 
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None

        with localconverter(ro.default_converter + pandas2ri.converter):
            sequence_cluster_df_r = ro.conversion.py2rpy(sequence_cluster_df)

        res = r['aovperm'](ro.Formula(f'`{self.evaluation_metric_field}` ~ C({CLUSTER_FIELD_NAME_STR})'),
                                      data=sequence_cluster_df_r,
                                      np=OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES)

        p_value = res.rx2('table').rx2('parametric P(>F)')[0]
        p_value_perm = res.rx2('table').rx2('resampled P(>F)')[0]

        return p_value, p_value_perm

    def _return_measure_association_aov(self,
                                        sequence_cluster_df: pd.DataFrame,
                                        method: AOVMeasueAssociationEnum) -> float:

        match method:
            case AOVMeasueAssociationEnum.ETA_SQUARED:
                measure_of_association = self._return_eta_squared(sequence_cluster_df)
            case AOVMeasueAssociationEnum.COHENS_F:
                measure_of_association = self._return_cohens_f(sequence_cluster_df)
            case AOVMeasueAssociationEnum.OMEGA_SQUARED:
                measure_of_association = self._return_omega_squared(sequence_cluster_df)
            case _:
                raise ValueError(OMNIBUS_TESTS_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{method}')

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
    
    def _return_omega_squared(self,
                              sequence_cluster_df: pd.DataFrame) -> float:
        """
        Although omega squared can theoretically be negative, here the minimum value will be set to 0, which means the
        absence of any effect. This is consistent with the interpretation of omega squared as a measure of effect size.
        ---
        [1] W. L. Hays, Statistics for Psychologists. New York: Holt, Rinehart and Winston, 1963.
        [2] A. D. A. Kroes and J. R. Finley, “Demystifying omega squared: Practical guidance for effect size in common analysis of variance designs.,” Psychological Methods, Jul. 2023, doi: 10.1037/met0000581.
        """

        n_groups = sequence_cluster_df[CLUSTER_FIELD_NAME_STR].nunique()
        df_within = sequence_cluster_df.shape[0] - n_groups

        grand_mean = sequence_cluster_df[self.evaluation_metric_field].mean()
        ss_between = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                         .transform(lambda x: (np.mean(x) - grand_mean)**2).sum())
        ss_within = (sequence_cluster_df.groupby(CLUSTER_FIELD_NAME_STR)[self.evaluation_metric_field]
                                        .transform(lambda x: (x - np.mean(x))**2).sum())
        
        mean_squared_error = ss_within / df_within
        
        omega_squared = (ss_between - (n_groups - 1) * mean_squared_error) / (ss_within + ss_between + mean_squared_error)

        # force omega squared to be non-negative(would introduce slight bias)
        # omega_squared = max(omega_squared, 0.0)
    
        return omega_squared

    def _return_measure_association_aov_conf_interval_bootstrap(self,
                                                                sequence_cluster_df: pd.DataFrame,
                                                                method: AOVMeasueAssociationEnum) -> tuple[Any, DefaultDict[str, int]]:

        measure_association_fail_dict = defaultdict(int)
        measure_association_fail_dict[method.value]
        def return_measure_association_bootstrap(x: np.ndarray, 
                                                 y: np.ndarray) -> float:

            sequence_cluster_df = pd.DataFrame({CLUSTER_FIELD_NAME_STR: x,
                                                self.evaluation_metric_field: y})

            measure_of_association = self._return_measure_association_aov(sequence_cluster_df,
                                                                          method)

            if np.isnan(measure_of_association):
                measure_association_fail_dict[method.value] += 1
                return 0
            else:
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

        return bootstrap_result, measure_association_fail_dict

    def _return_measure_association_aov_strength_value(self,
                                                       measure_of_association_value: float,
                                                       strength_guideline_method: AOVMeasureAssociationStrengthGuidelineEnum) -> str:

        match strength_guideline_method:
            case AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988:
                measure_of_association_strength = self._return_cohen_1988_measure_association_aov_strength(measure_of_association_value)
            case AOVMeasureAssociationStrengthGuidelineEnum.COHEN_1988_F:
                measure_of_association_strength = self._return_cohen_1988_f_measure_association_aov_strength(measure_of_association_value)
            case _:
                raise ValueError(OMNIBUS_TESTS_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR + f'{strength_guideline_method}')

        return measure_of_association_strength
    
    def _return_cohen_1988_measure_association_aov_strength(self,
                                                            measure_of_association_value: float) -> str:

        moa_guideline = Cohen1988MeasureAssociationStrengthAOV()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)

        if moa_strength is None:
            print(measure_of_association_value)
        
        return moa_strength.value

    def _return_cohen_1988_f_measure_association_aov_strength(self,
                                                              measure_of_association_value: float) -> str:

        moa_guideline = Cohen1988FMeasureAssociationStrengthAOV()
        moa_strength = moa_guideline.return_moa_strength(measure_of_association_value)
        
        return moa_strength.value

    def _return_measure_association_aov_results(self,
                                                sequence_cluster_df: pd.DataFrame) -> tuple[list[MeasureAssociationAOVResults], list[DefaultDict[str, int]]]:

        measure_of_association_aov_results_list = []
        measure_of_association_aov_fail_dict_list = []
        for moa_method, moa_strength_guide_method_list in OMNIBUS_TESTS_AOV_MEASURE_OF_ASSOCIATION_DICT.items():

            measure_of_association_type = moa_method.value
            measure_of_association_value = self._return_measure_association_aov(sequence_cluster_df,
                                                                                moa_method)

            moa_strength_guide_methods = [moa_strength_guide_method.value for moa_strength_guide_method in moa_strength_guide_method_list]
            moa_strength_guide_values = [self._return_measure_association_aov_strength_value(measure_of_association_value, 
                                                                                             moa_strength_guide_method)
                                            for moa_strength_guide_method in moa_strength_guide_method_list]

            bootstrap_result, measure_association_fail_dict = self._return_measure_association_aov_conf_interval_bootstrap(sequence_cluster_df,
                                                                                                                           moa_method)

            moa_strength_guide_conf_int_lower_bound_values = [self._return_measure_association_aov_strength_value(bootstrap_result.confidence_interval[0], 
                                                                                                                  moa_strength_guide_method)
                                                              for moa_strength_guide_method in moa_strength_guide_method_list]

            moa_strength_guide_conf_int_upper_bound_values = [self._return_measure_association_aov_strength_value(bootstrap_result.confidence_interval[1], 
                                                                                                                  moa_strength_guide_method)
                                                              for moa_strength_guide_method in moa_strength_guide_method_list]

            measure_of_association_aov_results = MeasureAssociationAOVResults(measure_of_association_type,
                                                                              measure_of_association_value,
                                                                              OMNIBUS_TESTS_BOOTSTRAPPING_CONFIDENCE_LEVEL,
                                                                              bootstrap_result.confidence_interval,
                                                                              bootstrap_result.bootstrap_distribution,
                                                                              bootstrap_result.standard_error,
                                                                              moa_strength_guide_methods,
                                                                              moa_strength_guide_values,
                                                                              moa_strength_guide_conf_int_lower_bound_values,
                                                                              moa_strength_guide_conf_int_upper_bound_values)

            measure_of_association_aov_results_list.append(measure_of_association_aov_results)
            measure_of_association_aov_fail_dict_list.append(measure_association_fail_dict)

        return measure_of_association_aov_results_list, measure_of_association_aov_fail_dict_list

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
                             OMNIBUS_TESTS_NUMBER_OF_OBSERVATIONS_FIELD_NAME_STR: anova_test_results.n_observations,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_F_TEST_STATISTIC_FIELD_NAME_STR: anova_test_results.f_statistic,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_DOF_BETWEEN_FIELD_NAME_STR: anova_test_results.degrees_of_freedom_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_DOF_WITHIN_FIELD_NAME_STR: anova_test_results.degrees_of_freedom_within,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_SS_BETWEEN_FIELD_NAME_STR: anova_test_results.ss_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_SS_WITHIN_FIELD_NAME_STR: anova_test_results.ss_within,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_MSS_BETWEEN_FIELD_NAME_STR: anova_test_results.mss_between,
                             OMNIBUS_TESTS_CONTINUOUS_AOV_MSS_WITHIN_FIELD_NAME_STR: anova_test_results.mss_within,
                             OMNIBUS_TESTS_PVAL_FIELD_NAME_STR: anova_test_results.p_value,
                             OMNIBUS_TESTS_PERM_PVAL_FIELD_NAME_STR: p_val_perm,
                             OMNIBUS_TESTS_R_PVAL_FIELD_NAME_STR: p_val_r,
                             OMNIBUS_TESTS_R_PERM_PVAL_FIELD_NAME_STR: p_val_r_perm,
                             OMNIBUS_TESTS_PERM_N_PERMS_FIELD_NAME_STR: OMNIBUS_TESTS_CONTINGENCY_PERMUTATION_N_RESAMPLES}

        measure_of_association_result_dict = self._return_measure_of_association_result_dict(measure_of_association_aov_results_list)

        test_results_dict = test_results_dict | measure_of_association_result_dict

        test_results_df = pd.DataFrame(test_results_dict,
                                       index = (0,))
        return test_results_df
    
    def _return_measure_of_association_result_dict(self,
                                                   measure_of_association_results_list: list[MeasureAssociationContingencyResults | MeasureAssociationAOVResults]) -> dict:

        measure_of_association_result_dict = {}
        for result in measure_of_association_results_list:

            measure_of_association_value_field_name = result.measure_type + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_VALUE_FIELD_NAME_STR
            measure_of_association_conf_int_value_field_name = result.measure_type + OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_VALUE_FIELD_NAME_STR

            measure_of_association_result = {measure_of_association_value_field_name: result.measure_value,
                                             measure_of_association_conf_int_value_field_name: [tuple(map(lambda x: round(x, OMNIBUS_TEST_RESULTS_ROUND_N_DIGITS), result.conf_int))]}

            moa_interpretation_guidelines_result = {}
            moa_interpretation_guidelines = zip(result.interpretation_guideline_methods, 
                                                result.interpretation_guideline_strength_values,
                                                result.interpretation_guideline_strength_for_conf_int_lower_bound_values,
                                                result.interpretation_guideline_strength_for_conf_int_upper_bound_values)

            for (moa_guideline_method, 
                 moa_guideline_strength_value, 
                 moa_guideline_strength_conf_int_lower_bound_value, 
                 moa_guideline_strength_conf_int_upper_bound_value) in moa_interpretation_guidelines:

                measure_of_association_interpretation_guideline_field_name = (result.measure_type +
                                                                              '_' + 
                                                                              moa_guideline_method + 
                                                                              OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_FIELD_NAME_STR)
                measure_of_association_interpretation_guideline_ci_lower_field_name = (measure_of_association_interpretation_guideline_field_name +
                                                                                       OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_LOWER_FIELD_NAME_STR)
                measure_of_association_interpretation_guideline_ci_upper_field_name = (measure_of_association_interpretation_guideline_field_name +
                                                                                       OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_STRENGTH_GUIDELINE_CONF_INT_UPPER_FIELD_NAME_STR)

                moa_interpretation_guidelines_result[measure_of_association_interpretation_guideline_field_name] = moa_guideline_strength_value
                moa_interpretation_guidelines_result[measure_of_association_interpretation_guideline_ci_lower_field_name] = moa_guideline_strength_conf_int_lower_bound_value
                moa_interpretation_guidelines_result[measure_of_association_interpretation_guideline_ci_upper_field_name] = moa_guideline_strength_conf_int_upper_bound_value
            
            measure_of_association_result = measure_of_association_result | moa_interpretation_guidelines_result
            
            measure_of_association_result_dict = measure_of_association_result_dict | measure_of_association_result

        additional_info_dict = {OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_ALPHA_FIELD_NAME_STR: result.conf_int_level,
                                OMNIBUS_TESTS_MEASURE_OF_ASSOCIATION_CONF_INT_N_BOOTSTRAP_SAMPLES_FIELD_NAME_STR: OMNIBUS_TESTS_BOOTSTRAPPING_EFFECT_SIZE_N_RESAMPLES}

        measure_of_association_result_dict = measure_of_association_result_dict | additional_info_dict

        return measure_of_association_result_dict

    def _perform_p_value_correction(self,
                                    test_results_df: pd.DataFrame) -> None:

        idx_to_insert = test_results_df.columns.get_loc(self._p_val_field_name_list[-1]) + 1

        for correction in OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_LIST:

            for field in self._p_val_field_name_list:

                has_none = test_results_df[field].isnull().any()
                if has_none:
                    corrected_p_values = None
                else:
                    corrected_p_values = sm.stats.multipletests(test_results_df[field], 
                                                                alpha=OMNIBUS_TESTS_ALPHA_LEVEL, 
                                                                method=correction.value)[1]

                label = field + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + correction.value 
                
                test_results_df.insert(idx_to_insert,
                                       label,
                                       corrected_p_values)

                idx_to_insert += 1

    def _add_p_value_is_significant_fields(self,
                                           test_results_df: pd.DataFrame) -> None:

        idx_to_insert = test_results_df.columns.get_loc(OMNIBUS_TESTS_PERM_N_PERMS_FIELD_NAME_STR) + 1

        field_list = self._p_val_field_name_list.copy()
        for correction in OMNIBUS_TESTS_P_VALUE_CORRECTION_METHOD_LIST:

            for field in self._p_val_field_name_list:

                label = field + OMNIBUS_TESTS_PVAL_CORRECTED_FIELD_NAME_STR + correction.value 
                field_list.append(label)
        
        for field in field_list:

            has_none = test_results_df[field].isnull().any()
            if has_none:
                is_significant = None
            else:
                is_significant = (test_results_df[field] <= OMNIBUS_TESTS_ALPHA_LEVEL).values
            
            label = field + OMNIBUS_TESTS_PVAL_IS_SIGNIFICANT_FIELD_NAME_STR

            test_results_df.insert(idx_to_insert,
                                   label,
                                   is_significant)

            idx_to_insert += 1
    
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