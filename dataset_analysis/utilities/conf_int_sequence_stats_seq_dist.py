from .configs.general_config import *
from .configs.conf_int_config import *
from .constants.constants import *
from .standard_import import *
from .sequence_distance_analysis import SequenceDistanceAnalytics
from .sequence_statistics_functions import SequenceStatistics
from .data_classes import *
from .validators import *
from .result_tables import ResultTables


class ConfIntSequenceStatsSeqDist:

    def __init__(
        self,
        dataset_name: str,
        sequence_statistics: SequenceStatistics,
        sequence_distance_analytics: SequenceDistanceAnalytics,
    ):

        self.dataset_name = dataset_name
        self.sequence_statistics = sequence_statistics
        self.sequence_distance_analytics = sequence_distance_analytics

        self.groups = self._return_groups(
            self.sequence_statistics.unique_learning_activity_sequence_stats_per_group
        )

        self.sequence_statistics_conf_int_df = None

    def calculate_confidence_intervals(self) -> None:

        param_list = self._return_param_list()
        len_param_list = len(param_list)
        param_data_generator = self._return_param_and_data_generator(param_list)

        results = Parallel(n_jobs=NUMBER_OF_CORES)(
            delayed(self._return_conf_int_result_dict)(
                sequence_type, statistic, estimator, group, data
            )
            for sequence_type, statistic, estimator, group, data in tqdm(
                param_data_generator, desc="Processing", total=len_param_list
            )
        )

        sort_list = [
            ConfIntResultFields.DATASET_NAME.value,
            ConfIntResultFields.SEQUENCE_TYPE.value,
            ConfIntResultFields.SEQUENCE_STATISTIC.value,
            ConfIntResultFields.ESTIMATOR.value,
            ConfIntResultFields.GROUP.value,
        ]
        self.sequence_statistics_conf_int_df = (
            pd.DataFrame(results).sort_values(by=sort_list).reset_index(drop=True)
        )

    def add_confidence_intervals_to_result_tables(
        self, result_tables: ResultTables
    ) -> None:

        _ = check_value_not_none(
            self.sequence_statistics_conf_int_df,
            CONFIDENCE_INTERVAL_ERROR_NO_RESULTS_NAME_STR,
        )
        # add data to results_table
        result_tables.seq_stat_conf_int_df = self.sequence_statistics_conf_int_df.copy()

    def _calculate_confidence_interval(
        self,
        statistic: SequenceStatistic,
        estimator: ConfIntEstimator,
        data: np.ndarray,
    ) -> SequenceStatisticConfIntResult:

        estimator_function = self._return_bootstrap_estimator(estimator)

        match statistic:
            case (
                SequenceStatistic.SEQUENCE_LENGTH
                | SequenceStatistic.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ
                | SequenceStatistic.PCT_REPEATED_LEARNING_ACTIVITIES
            ):
                statistic_value = estimator_function(data)
                ci_lower_bound, ci_upper_bound, bootstrap_se = (
                    self._return_bootstrap_conf_int_seq_stat(
                        data,
                        estimator_function,
                        CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL,
                    )
                )

            case (
                SequenceStatistic.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):
                averaging_method = self._return_sequence_distance_averaging_method(
                    SequenceDistanceAveragingMethod.MEAN
                )

                calculate_statistic_for_average_distances = (
                    self._return_calculate_statistic_for_average_distances(
                        data,
                        estimator_function,
                        averaging_method,
                        SelfDistanceFilter.MAIN_DIAGONAL,
                    )
                )
                index = np.arange(data.shape[0])
                statistic_value = calculate_statistic_for_average_distances(index)

                ci_lower_bound, ci_upper_bound, bootstrap_se = (
                    self._return_bootstrap_conf_int_seq_dist(
                        data,
                        estimator_function,
                        averaging_method,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_SEQ_DIST_SELF_DISTANCE_FILTER,
                        CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL,
                    )
                )

            case (
                SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):
                averaging_method = self._return_sequence_distance_averaging_method(
                    SequenceDistanceAveragingMethod.MEDIAN
                )

                calculate_statistic_for_average_distances = (
                    self._return_calculate_statistic_for_average_distances(
                        data,
                        estimator_function,
                        averaging_method,
                        SelfDistanceFilter.MAIN_DIAGONAL,
                    )
                )
                index = np.arange(data.shape[0])
                statistic_value = calculate_statistic_for_average_distances(index)

                ci_lower_bound, ci_upper_bound, bootstrap_se = (
                    self._return_bootstrap_conf_int_seq_dist(
                        data,
                        estimator_function,
                        averaging_method,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_SEQ_DIST_SELF_DISTANCE_FILTER,
                        CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD,
                        CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL,
                    )
                )

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceStatistic.__name__}"
                )

        conf_int_result = SequenceStatisticConfIntResult(
            statistic_value,
            ci_lower_bound,
            ci_upper_bound,
            CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD,
            CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL,
            CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES,
            bootstrap_se,
        )

        return conf_int_result

    def _return_data(
        self, sequence_type: SequenceType, statistic: SequenceStatistic, group: int
    ) -> np.ndarray:

        match statistic:
            case (
                SequenceStatistic.SEQUENCE_LENGTH
                | SequenceStatistic.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ
                | SequenceStatistic.PCT_REPEATED_LEARNING_ACTIVITIES
            ):

                data = self._return_seq_stat_data_by_sequence_type(
                    sequence_type, statistic, group
                )

            case (
                SequenceStatistic.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):

                data = self._return_seq_dist_data_by_sequence_type(
                    sequence_type, SequenceDistanceType.NON_NORMALIZED, group
                )

            case (
                SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):

                data = self._return_seq_dist_data_by_sequence_type(
                    sequence_type, SequenceDistanceType.NORMALIZED, group
                )

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceStatistic.__name__}"
                )

        return data

    def _return_seq_stat_data_by_sequence_type(
        self, sequence_type: SequenceType, statistic: SequenceStatistic, group: int
    ) -> np.ndarray:

        match sequence_type:
            case SequenceType.ALL_SEQUENCES:
                data_for_sequence_type = (
                    self.sequence_statistics.learning_activity_sequence_stats_per_group
                )

            case SequenceType.UNIQUE_SEQUENCES:
                data_for_sequence_type = (
                    self.sequence_statistics.unique_learning_activity_sequence_stats_per_group
                )

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceType.__name__}"
                )

        group_filter = data_for_sequence_type[GROUP_FIELD_NAME_STR] == group
        data_for_sequence_type = (
            data_for_sequence_type.loc[group_filter, statistic.value].copy().to_numpy()
        )

        return data_for_sequence_type

    def _return_seq_dist_data_by_sequence_type(
        self,
        sequence_type: SequenceType,
        sequence_dist_type: SequenceDistanceType,
        group: int,
    ) -> np.ndarray:

        match sequence_type:
            case SequenceType.ALL_SEQUENCES:
                use_unique_sequence_distances = False

            case SequenceType.UNIQUE_SEQUENCES:
                use_unique_sequence_distances = True

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceType.__name__}"
                )

        match sequence_dist_type:
            case SequenceDistanceType.NON_NORMALIZED:
                normalize_sequence_distance = False

            case SequenceDistanceType.NORMALIZED:
                normalize_sequence_distance = True

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceDistanceType.__name__}"
                )

        data_for_sequence_type = (
            self.sequence_distance_analytics.return_sequence_distance_matrix_per_group(
                group, normalize_sequence_distance, use_unique_sequence_distances
            )[SEQUENCE_DISTANCE_ANALYTICS_SEQUENCE_DIST_SQUARE_MATRIX_FIELD_NAME_STR]
            .copy()
            .to_numpy()
        )

        return data_for_sequence_type

    def _return_bootstrap_estimator(
        self,
        estimator: ConfIntEstimator,
    ) -> Callable[[np.ndarray | List], float]:

        match estimator:
            case ConfIntEstimator.MEAN:
                estimator = np.mean

            case ConfIntEstimator.MEDIAN:
                estimator = np.median

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{ConfIntEstimator.__name__}"
                )

        return estimator

    def _return_sequence_distance_averaging_method(
        self,
        seq_dist_averaging_method: SequenceDistanceAveragingMethod,
    ) -> Callable[[np.ndarray | List], float]:

        match seq_dist_averaging_method:
            case SequenceDistanceAveragingMethod.MEAN:
                seq_dist_averaging_function = np.mean

            case SequenceDistanceAveragingMethod.MEDIAN:
                seq_dist_averaging_function = np.median

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceDistanceAveragingMethod.__name__}"
                )

        return seq_dist_averaging_function

    def _return_conf_int_result_dict(
        self,
        sequence_type: SequenceType,
        statistic: SequenceStatistic,
        estimator: ConfIntEstimator,
        group: int,
        data: np.ndarray,
    ) -> Dict[str, str | int | float | SequenceStatisticConfIntResult]:

        seq_stat_conf_int_result = self._calculate_confidence_interval(
            statistic, estimator, data
        )

        seq_stat_conf_int_control_vars = self._return_conf_int_control_vars(
            statistic, data
        )

        conf_int_result_dict = {
            ConfIntResultFields.DATASET_NAME.value: self.dataset_name,
            ConfIntResultFields.SEQUENCE_TYPE.value: sequence_type.value,
            ConfIntResultFields.SEQUENCE_STATISTIC.value: statistic.value,
            ConfIntResultFields.ESTIMATOR.value: estimator.value,
            ConfIntResultFields.BOOTSTRAP_METHOD.value: CONFIDENCE_INTERVAL_BOOTSTRAP_METHOD,
            ConfIntResultFields.CONFIDENCE_LEVEL.value: CONFIDENCE_INTERVAL_BOOTSTRAP_CONFIDENCE_LEVEL,
            ConfIntResultFields.N_BOOTSTRAP_RESAMPLES.value: CONFIDENCE_INTERVAL_N_BOOTSTRAP_SAMPLES,
            ConfIntResultFields.GROUP.value: group,
            ConfIntResultFields.STATISTIC_VALUE.value: seq_stat_conf_int_result.statistic_value,
            ConfIntResultFields.LOWER_BOUND.value: seq_stat_conf_int_result.conf_int_lower_bound,
            ConfIntResultFields.UPPER_BOUND.value: seq_stat_conf_int_result.conf_int_upper_bound,
            ConfIntResultFields.SAMPLE_SIZE.value: seq_stat_conf_int_control_vars.sample_size,
            ConfIntResultFields.N_UNIQUE_VALUES.value: seq_stat_conf_int_control_vars.n_unique_values,
            ConfIntResultFields.VARIANCE.value: seq_stat_conf_int_control_vars.variance,
            ConfIntResultFields.STANDARD_DEVIATION.value: seq_stat_conf_int_control_vars.standard_deviation,
            ConfIntResultFields.BOOTSTRAP_STANDARD_ERROR.value: seq_stat_conf_int_result.bootstrap_standard_error,
            ConfIntResultFields.RESULT.value: seq_stat_conf_int_result,
        }

        return conf_int_result_dict

    def _return_param_list(
        self,
    ) -> List[Tuple[SequenceType, SequenceStatistic, ConfIntEstimator, int]]:

        conf_int_type_iterator = list(
            product(
                SEQUENCE_TYPE_LIST, SEQUENCE_STATISTIC_LIST, ESTIMATOR_LIST, self.groups
            )
        )

        conf_int_type_iterator = [
            (seq_type, seq_stat, estimator, group)
            for seq_type, seq_stat, estimator, group in conf_int_type_iterator
            if self._filter_conf_int_type(seq_type, seq_stat)
        ]

        return conf_int_type_iterator

    def _return_param_and_data_generator(self, param_iterator: List) -> Generator[
        Tuple[SequenceType, SequenceStatistic, ConfIntEstimator, int, np.ndarray],
        None,
        None,
    ]:

        for sequence_type, statistic, estimator, group in param_iterator:
            data = self._return_data(sequence_type, statistic, group)
            yield sequence_type, statistic, estimator, group, data

    def _return_bootstrap_conf_int_seq_stat(
        self,
        sequence_stat_array: np.ndarray,
        estimator: Callable[[np.ndarray | List], float],
        number_bootstrap_resamples: int,
        bootstrap_method: str,
        confidence_level: float,
    ) -> Tuple[float, float, float]:
        """Return the bootstrap confidence interval of the specified estimator for a sequence statistic."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateDataWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            bootstrap_result = bootstrap(
                (sequence_stat_array,),
                estimator,
                n_resamples=number_bootstrap_resamples,
                method=bootstrap_method,
                confidence_level=confidence_level,
                vectorized=False,
                paired=False,
                alternative="two-sided",
            )

        return (
            bootstrap_result.confidence_interval.low,
            bootstrap_result.confidence_interval.high,
            bootstrap_result.standard_error,
        )

    def _return_bootstrap_conf_int_seq_dist(
        self,
        distance_matrix: np.ndarray,
        estimator: Callable[[np.ndarray | List], float],
        averaging_method: Callable[[np.ndarray | List], float],
        seq_filter: SelfDistanceFilter,
        number_bootstrap_resamples: int,
        bootstrap_method: str,
        confidence_level: float,
    ) -> Tuple[float, float, float]:
        """Return the bootstrap confidence interval of the specified estimator for an user-wise averaged sequence distance.
        The averaging method should be one of [mean, median]. In resampling a new distance matrix is created for only
        the users in the bootstrap sample in order to prevent a biased bootstrap distribution.
        """

        calculate_statistic_for_average_distances = (
            self._return_calculate_statistic_for_average_distances(
                distance_matrix, estimator, averaging_method, seq_filter
            )
        )

        index = np.arange(distance_matrix.shape[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateDataWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            bootstrap_result = bootstrap(
                (index,),
                calculate_statistic_for_average_distances,
                n_resamples=number_bootstrap_resamples,
                method=bootstrap_method,
                confidence_level=confidence_level,
                vectorized=False,
                paired=False,
                alternative="two-sided",
            )

        return (
            bootstrap_result.confidence_interval.low,
            bootstrap_result.confidence_interval.high,
            bootstrap_result.standard_error,
        )

    def _return_calculate_statistic_for_average_distances(
        self,
        distance_matrix: np.ndarray,
        estimator: Callable[[np.ndarray | List], float],
        averaging_method: Callable[[np.ndarray | List], float],
        seq_filter: SelfDistanceFilter,
    ) -> Callable[[np.ndarray], float]:
        def calculate_statistic_for_average_distances(row_indices: np.ndarray) -> float:
            dist_mat_index_filtered = distance_matrix[row_indices][:, row_indices]

            dist_mat_size = dist_mat_index_filtered.shape[0]
            index = np.arange(dist_mat_size)

            average_distances = []
            for i, (row_index, row) in enumerate(
                zip(row_indices, dist_mat_index_filtered)
            ):
                match seq_filter:
                    case SelfDistanceFilter.NO_FILTER:
                        filter_array = np.full(dist_mat_size, True)

                    case SelfDistanceFilter.MAIN_DIAGONAL:
                        filter_array = index != i

                    case SelfDistanceFilter.ALL_SELF_DISTANCES:
                        filter_array = row_indices != row_index

                    case _:
                        raise ValueError(
                            "Not a valid member of enum: "
                            + f"{SelfDistanceFilter.__name__}"
                        )

                average_distance = averaging_method(row[filter_array].flatten())
                average_distances.append(average_distance)
            statistic = estimator(average_distances)

            return statistic

        return calculate_statistic_for_average_distances

    def _return_groups(
        self,
        sequence_statistics: pd.DataFrame,
    ) -> List[int]:

        return list(sequence_statistics[GROUP_FIELD_NAME_STR].unique())

    def _return_conf_int_control_vars(
        self, statistic: SequenceStatistic, data: np.ndarray
    ) -> SequenceStatisticConfIntDataControlVars:

        match statistic:
            case (
                SequenceStatistic.SEQUENCE_LENGTH
                | SequenceStatistic.PCT_UNIQUE_LEARNING_ACTIVITIES_PER_GROUP_IN_SEQ
                | SequenceStatistic.PCT_REPEATED_LEARNING_ACTIVITIES
            ):

                pass

            case (
                SequenceStatistic.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):
                averaging_method = self._return_sequence_distance_averaging_method(
                    SequenceDistanceAveragingMethod.MEAN
                )

                data = averaging_method(data, axis=1)

            case (
                SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
            ):
                averaging_method = self._return_sequence_distance_averaging_method(
                    SequenceDistanceAveragingMethod.MEDIAN
                )

                data = averaging_method(data, axis=1)

            case _:
                raise ValueError(
                    CONFIDENCE_INTERVAL_ERROR_ENUM_NON_VALID_MEMBER_NAME_STR
                    + f"{SequenceStatistic.__name__}"
                )

        sample_size = data.shape[0]
        n_unique_vals = len(np.unique(data))
        var = np.var(data, ddof=1)
        std = np.std(data, ddof=1)

        data_control_vars = SequenceStatisticConfIntDataControlVars(
            sample_size, n_unique_vals, var, std
        )

        return data_control_vars

    def _filter_conf_int_type(
        self,
        sequence_type: SequenceType,
        sequence_statistic: SequenceStatistic,
    ) -> bool:

        match sequence_type:
            case SequenceType.ALL_SEQUENCES:
                match sequence_statistic:
                    case (
                        SequenceStatistic.MEAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                        | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                        | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                        | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_UNIQUE_SEQUENCES
                    ):
                        return False

                    case _:
                        return True

            case SequenceType.UNIQUE_SEQUENCES:
                match sequence_statistic:
                    case (
                        SequenceStatistic.MEAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                        | SequenceStatistic.MEDIAN_SEQUENCE_DISTANCE_ALL_SEQUENCES
                        | SequenceStatistic.MEAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                        | SequenceStatistic.MEDIAN_NORMALIZED_SEQUENCE_DISTANCE_ALL_SEQUENCES
                    ):
                        return False

                    case _:
                        return True
