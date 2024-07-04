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
    seq_count_per_cluster: pd.DataFrame

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