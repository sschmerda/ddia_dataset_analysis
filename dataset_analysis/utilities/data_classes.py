from .standard_import import *

@dataclass
class HeatmapAnnotationVars:
    masked_data_df: pd.DataFrame
    masked_data: Any
    optimum_values_positions: List[tuple]
    optimum_values_idx: int

@dataclass
class AlgoParams:
    cluster_algo_name: str
    dim_reduction_algo_name: str | None
    cluster_param_series: pd.Series
    dim_reduction_param_series: pd.Series | None

@dataclass
class ClusterValidationResults:
    cluster_validation_metric_name: str
    cluster_validation_metric_optimum_value: float
    number_clusters: int
    percentage_clustered: float

@dataclass
class ClusterResults:
    cluster_labels: NDArray[np.int_]
    cluster_entity_ids: NDArray[np.int_]
    clustered: NDArray[np.int_]
    number_clusters: int
    percentage_clustered: float
    min_cluster_size: int
    sequence_distances_is_normalized: bool
    cluster_entity_type: str
    best_dim_reduction_parameters: dict

@dataclass
class TestResultsChiSquared:
    observed_frequency: np.ndarray
    expected_frequency: np.ndarray
    n_observations: int
    chi_squared_statistic: float
    degrees_of_freedom: int
    p_value: float

@dataclass
class ContingencyExpectedFrequenciesStats:
    expected_frequencies_threshold: int
    has_expected_frequency_below_threshold: bool
    n_elements_contingency_table: int
    n_elements_contingency_table_expected_below_threshold: int
    pct_elements_contingency_table_expected_below_threshold: float
    table_dimensions: tuple[int, int]

@dataclass
class MeasureAssociationContingencyResults:
    measure_type: str
    measure_value: float
    conf_int_level: float
    conf_int: tuple[float, float]
    bootstrap_dist: np.ndarray
    bootstrap_standard_error: float | np.ndarray
    interpretation_guideline_methods: list[str]
    interpretation_guideline_strength_values: list[str]

@dataclass
class TestResultsAOV:
    n_observations: int
    f_statistic: float
    degrees_of_freedom_between: int
    degrees_of_freedom_within: int
    ss_between: int
    ss_within: int
    mss_between: float
    mss_within: float
    p_value: float

@dataclass
class MeasureAssociationAOVResults:
    measure_type: str
    measure_value: float
    conf_int_level: float
    conf_int: tuple[float, float]
    bootstrap_dist: np.ndarray
    bootstrap_standard_error: float | np.ndarray
    interpretation_guideline_methods: list[str]
    interpretation_guideline_strength_values: list[str]

@dataclass
class OmnibusTestResults:
    group: int
    test_result_df: pd.DataFrame 
    measure_of_association_results: list[MeasureAssociationContingencyResults] | list[MeasureAssociationAOVResults]
    measure_of_association_fail_dict: list[DefaultDict[str, int]]