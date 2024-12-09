from .configs.general_config import *
from .constants.constants import *
from .standard_import import *


def keep_last_repeated_learning_activities(
    interactions: pd.DataFrame,
    group_field: str,
    user_field: str | None,
    learning_activity_field: str,
    timestamp_field: str,
):
    """Filters out all but the last of repeated learning activities in the interactions dataframe per user-group sequence

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        Can be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning_activity field column
    timestamp_field : str
        The timestamp field column

    Returns
    -------
    pd.DataFrame
        The filtered dataframe
    """
    sort_list = [group_field, user_field, timestamp_field]
    sort_list = [i for i in sort_list if i]
    interactions = interactions.reset_index(drop=True)
    interactions = interactions.sort_values(by=sort_list)
    initial_len = interactions.shape[0]

    keep_index_list = []

    if group_field:
        group_list = [group_field, user_field]
    else:
        group_list = user_field

    for _, df in tqdm(interactions.groupby(group_list)):
        la_prev = None

        for index, learning_activity in df[learning_activity_field].iloc[::-1].items():
            if learning_activity == la_prev:
                continue
            else:
                keep_index_list.append(index)

            la_prev = learning_activity

    interactions = interactions.loc[keep_index_list]
    interactions = interactions.sort_values(by=sort_list)

    final_len = interactions.shape[0]
    n_removed_interactions = initial_len - final_len
    n_removed_interactions_pct = (initial_len - final_len) / initial_len * 100

    print(STAR_STRING)
    print("\n")
    print(f"= Repeated Consecutive Learning Activity Removal =")
    print("\n")
    print(f"Initial number of {ROWS_NAME_STR.lower()}: {initial_len}")
    print(f"Final number of {ROWS_NAME_STR.lower()}: {final_len}")
    print(f"Removed number of {ROWS_NAME_STR.lower()}: {n_removed_interactions}")
    print(
        f"Removed percentage of {ROWS_NAME_STR.lower()}: {n_removed_interactions_pct}%"
    )
    print("\n")
    print(STAR_STRING)

    return interactions


def return_and_plot_evaluation_score_range(
    interactions: pd.DataFrame,
    learning_activity_field: str,
    evaluation_learning_activity_score_field: str,
    group_field: str | None,
    evaluation_group_score_field: str,
    evaluation_course_score_field: str,
    result_tables: Type[Any],
) -> None:
    """Returns and plots evaluation score ranges. The information can be used to determine a threshold level to convert a score into an is_correct evaluation field (add_evaluation_field function).
    Unavailable parameters can be set to None and will not be considered in the calculations.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    learning_activity_field : str
        The learning_activity field
    evaluation_learning_activity_score_field : str
        The learning_activity evaluation score field
    group_field : str
        The group field
        Can be set to None if the interactions dataframe does not have a group_field
    evaluation_group_score_field : str
        The group evaluation score field
    evaluation_course_score_field : str
        The course evaluation score field
    result_tables : Type[Any]
        The ResultTables object

    Returns
    -------
    None
    """

    # helper function - calculate min and max values for pointplot
    def return_min_and_max(array):
        minimum = min([el for el in array if el is not None])
        maximum = max([el for el in array if el is not None])

        return (minimum, maximum)

    eval_score_ranges_dict = {}
    for group, score, group_name in zip(
        [learning_activity_field, group_field, "course"],
        [
            evaluation_learning_activity_score_field,
            evaluation_group_score_field,
            evaluation_course_score_field,
        ],
        [LEARNING_ACTIVITY_FIELD_NAME_STR, GROUP_FIELD_NAME_STR, COURSE_FIELD_NAME_STR],
    ):
        eval_score_ranges_dict[group_name] = {
            "eval_score_ranges": None,
            "range_count_frequency": None,
        }
        if group and score:
            if (group == learning_activity_field) or (group == group_field):
                eval_score_ranges = interactions.groupby(group)[score].agg(
                    func=[min, max, np.mean, np.median]
                )
                eval_score_ranges = eval_score_ranges.rename(
                    columns={
                        "min": "score_minimum",
                        "max": "score_maximum",
                        "mean": "score_mean",
                        "median": "score_median",
                    }
                )
                eval_score_ranges = eval_score_ranges.reset_index()
                eval_score_ranges_long = pd.melt(
                    eval_score_ranges,
                    id_vars=[group],
                    value_vars=["score_minimum", "score_maximum"],
                    var_name="boundary_type",
                    value_name="boundary_value",
                ).sort_values(by=group)

            else:
                eval_score_ranges = interactions[score].agg(
                    func=[min, max, np.mean, np.median]
                )
                eval_score_ranges = (
                    pd.DataFrame(eval_score_ranges).transpose().reset_index()
                )
                eval_score_ranges = eval_score_ranges.rename(
                    columns={
                        "index": group,
                        "min": "score_minimum",
                        "max": "score_maximum",
                        "mean": "score_mean",
                        "median": "score_median",
                    }
                )
                eval_score_ranges_long = pd.melt(
                    eval_score_ranges,
                    id_vars=[group],
                    value_vars=["score_minimum", "score_maximum"],
                    var_name="boundary_type",
                    value_name="boundary_value",
                ).sort_values(by=group)
                eval_score_ranges_long[group] = [0, 0]

            range_series = pd.Series(
                eval_score_ranges[["score_minimum", "score_maximum"]].itertuples(
                    index=False, name=None
                )
            )
            range_count_df = (
                range_series.value_counts()
                .reset_index()
                .rename(columns={"index": "range", 0: "count"})
            )
            range_frequency_df = (
                range_series.value_counts(normalize=True)
                .reset_index()
                .rename(columns={"index": "range", 0: "frequency"})
            )
            range_counts_frequency_df = range_count_df.merge(
                range_frequency_df, how="left", on="range"
            )
            range_counts_frequency_df
            number_of_score_ranges = range_counts_frequency_df["range"].nunique()
            number_of_groups = eval_score_ranges[group].nunique()

            eval_score_ranges_dict[group_name] = {
                "eval_score_ranges": eval_score_ranges,
                "range_count_frequency": range_counts_frequency_df,
            }

            are_equal_all_score_minima = (
                eval_score_ranges["score_minimum"].nunique() == 1
            )
            are_equal_all_score_maxima = (
                eval_score_ranges["score_maximum"].nunique() == 1
            )
            score_minimum = float(eval_score_ranges["score_minimum"].min())
            score_maximum = float(eval_score_ranges["score_maximum"].max())

            print(STAR_STRING)
            print(f"{group_name} Evaluation Score Range:")
            print(STAR_STRING)
            print("")
            print(f"Number of {group_name}s: {number_of_groups}")
            print(f"Number of Unique Score Ranges: {number_of_score_ranges}")
            print(
                f"Minima of {group_name} Ranges are all equal: {are_equal_all_score_minima}"
            )
            print(
                f"Maxima of {group_name} Ranges are all equal: {are_equal_all_score_maxima}"
            )
            print(f"Minimum {group_name} Score: {score_minimum}")
            print(f"Maximum {group_name} Score: {score_maximum}")

            # scatter
            g = sns.JointGrid(
                data=eval_score_ranges, x="score_minimum", y="score_maximum"
            )
            g.plot_joint(sns.scatterplot, alpha=0.6, legend=False)
            g.plot_marginals(sns.rugplot, height=1, alpha=0.6)
            g.set_axis_labels(
                xlabel=f"{group_name} Minimum Score",
                ylabel=f"{group_name} Maximum Score",
            )
            plt.show()

            # barplot with min and max scores
            groups = sorted(
                list(np.unique(eval_score_ranges_long[group])), key=lambda x: int(x)
            )
            mins = [
                eval_score_ranges_long[eval_score_ranges_long[group] == g][
                    "boundary_value"
                ].min()
                for g in groups
            ]
            lengths = [
                eval_score_ranges_long[eval_score_ranges_long[group] == g][
                    "boundary_value"
                ].max()
                - g_mins
                for g, g_mins in zip(groups, mins)
            ]

            mean_marker = {
                "marker": "o",
                "markerfacecolor": "green",
                "markeredgecolor": "black",
                "markersize": 10,
            }

            median_marker = {"linewidth": 5, "linestyle": "-", "color": "blue"}

            g = sns.barplot(x=lengths, y=groups, left=mins, palette="turbo", orient="h")
            g = sns.stripplot(
                x="boundary_value",
                y=group,
                order=groups,
                color="red",
                s=10,
                edgecolor="black",
                alpha=1,
                linewidth=2,
                jitter=False,
                clip_on=False,
                data=eval_score_ranges_long,
                orient="h"
            )
            if (group == learning_activity_field) or (group == group_field):
                g = sns.boxplot(
                    showmeans=True,
                    meanline=False,
                    meanprops=mean_marker,
                    medianprops=median_marker,
                    whiskerprops={"visible": False},
                    x=score,
                    y=group,
                    order=groups,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    data=interactions,
                    orient="h"
                )
                g.set(xlabel=f"{group_name} Score Range", ylabel=f"{group_name}")
                plt.show()
            else:
                g = sns.boxplot(
                    showmeans=True,
                    meanline=False,
                    meanprops=mean_marker,
                    medianprops=median_marker,
                    whiskerprops={"visible": False},
                    x=score,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    data=interactions,
                    orient="h"
                )
                g.set(xlabel=f"{group_name} Score Range", ylabel=f"{group_name}")
                plt.show()

            # barplot with all scores
            g = sns.barplot(x=lengths, y=groups, left=mins, palette="turbo", orient="h")
            if (group == learning_activity_field) or (group == group_field):
                g = sns.stripplot(
                    x=score,
                    y=group,
                    order=groups,
                    color="white",
                    edgecolor="black",
                    linewidth=1.0,
                    jitter=False,
                    clip_on=False,
                    data=interactions,
                    orient="h"
                )
            else:
                g = sns.stripplot(
                    x=score,
                    color="white",
                    edgecolor="black",
                    linewidth=1.0,
                    jitter=False,
                    clip_on=False,
                    data=interactions,
                    orient="h"
                )

            if (group == learning_activity_field) or (group == group_field):
                g = sns.boxplot(
                    showmeans=True,
                    meanline=False,
                    meanprops=mean_marker,
                    medianprops=median_marker,
                    whiskerprops={"visible": False},
                    x=score,
                    y=group,
                    order=groups,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    data=interactions,
                    orient="h"
                )
                g.set(xlabel=f"{group_name} Score Range", ylabel=f"{group_name}")
                plt.show()
            else:
                g = sns.boxplot(
                    showmeans=True,
                    meanline=False,
                    meanprops=mean_marker,
                    medianprops=median_marker,
                    whiskerprops={"visible": False},
                    x=score,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    data=interactions,
                    orient="h"
                )
                g.set(xlabel=f"{group_name} Score", ylabel=f"{group_name}")
                plt.show()

            # pointplot
            if (group == learning_activity_field) or (group == group_field):
                g = sns.pointplot(
                    data=interactions,
                    x=score,
                    y=group,
                    order=groups,
                    errorbar=return_min_and_max,
                    capsize=0.4,
                    join=False,
                    orient="h"
                )
                g.set(xlabel=f"{group_name} Score Range", ylabel=f"{group_name}")
                plt.show()
            else:
                g = sns.pointplot(
                    data=interactions,
                    x=score,
                    errorbar=return_min_and_max,
                    capsize=0.4,
                    join=False,
                )
                g.set(xlabel=f"{group_name} Score Range", ylabel=f"{group_name}")
                plt.show()

            # score-ranges countplot
            g = sns.barplot(data=range_counts_frequency_df, x="count", y="range")
            g.set(xlabel=f"{group_name} Count", ylabel=f"{group_name} Score Range")
            plt.show()
            g = sns.barplot(data=range_counts_frequency_df, x="frequency", y="range")
            g.set(xlabel=f"{group_name} Frequency", ylabel=f"{group_name} Score Range")
            plt.show()

    # add data to results_table
    result_tables.eval_score_ranges_dict = eval_score_ranges_dict.copy()


def add_unique_identifier_code_to_learning_activity_in_sequence(
    interactions: pd.DataFrame,
    group_field: str,
    user_field: str,
    learning_activity_field: str,
    timestamp_field: str,
    order_field: str,
):
    """Add an identifier field to the interactions dataframe which maps an unique code to each uninterrupted learning activity sub-sequence with the same name.
    This is used for calculating evaluation metrics for the respective learning activity within a sequence of learning activities.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    timestamp_field : str
        The timestamp field column
    order_field : str
        The order field column

    Returns
    -------
    tuple
        A tuple containing the interactions dataframe with added identifier column and the name of the identifier column
    """
    sort_list = [group_field, user_field, timestamp_field, order_field]
    sort_list_filtered = [i for i in sort_list if i]

    interactions = interactions.sort_values(by=sort_list_filtered)

    learning_activities_list = interactions[learning_activity_field].to_list()

    code_list = []
    last_learning_activity = None
    code = 0

    for learning_activity in learning_activities_list:
        if learning_activity != last_learning_activity:
            code += 1
            code_list.append(code)
            last_learning_activity = learning_activity
        else:
            code_list.append(code)

    # learning activity code field name can be hardcoded because this field will be removed anyways
    learning_activity_code_field = "la_code_for_grouping"
    interactions[learning_activity_code_field] = code_list

    return (interactions, learning_activity_code_field)


def generate_evaluation_learning_activity_merge_dict(
    group_field: str,
    user_field: str,
    learning_activity_field: str,
    learning_activity_code_field: str,
):
    """Generates a dictionary which maps all evaluation metrics at learning activity level and the input arguments
    (identifier fields which identify the learning activity for which the evaluation metric is calculated) to an empty list.

    Parameters
    ----------
    group_field : str
        The group field column
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    learning_activity_code_field : str
        The learning activity code field column

    Returns
    -------
    dict
        The mapping dictionary
    """
    # evaluation_learning_activity_fields_list contains all evaluation metric fields at learning activity level and is defined in the constants.py module
    evaluation_learning_activity_merge_dict = {
        eval_metric: [] for eval_metric in EVALUATION_LEARNING_ACTIVITY_FIELD_LIST
    }

    if group_field:
        evaluation_learning_activity_merge_dict[group_field] = []
    evaluation_learning_activity_merge_dict[user_field] = []
    evaluation_learning_activity_merge_dict[learning_activity_field] = []
    evaluation_learning_activity_merge_dict[learning_activity_code_field] = []

    return evaluation_learning_activity_merge_dict


def fill_evaluation_learning_activity_merge_dict(
    evaluation_learning_activity_merge_dict,
    group_field=None,
    user_field=None,
    learning_activity_field=None,
    learning_activity_code_field=None,
    group=None,
    user=None,
    learning_activity=None,
    learning_activity_code=None,
    number_interactions_total=None,
    number_attempts_total=None,
    number_hints_total=None,
    time_total=None,
    single_score=None,
    single_score_hint_lowest=None,
    single_score_not_first_attempt_lowest=None,
    score_highest=None,
    score_highest_without_hint=None,
    score_first_attempt=None,
    score_last_attempt=None,
    number_interactions_until_score_highest=None,
    number_attempts_until_score_highest=None,
    number_hints_until_score_highest=None,
    time_until_score_highest=None,
    is_correct=None,
    is_correct_without_hint=None,
    is_correct_first_attempt=None,
    is_correct_first_attempt_without_hint=None,
    is_correct_last_attempt=None,
    is_correct_last_attempt_without_hint=None,
    number_interactions_until_correct=None,
    number_attempts_until_correct=None,
    number_hints_until_correct=None,
    time_until_correct=None,
):
    """Fills the lists of the evaluation_learning_activity_merge_dict with learning activity identification and evaluation metric values

    Parameters
    ----------
    evaluation_learning_activity_merge_dict : dict
        The dictionray which maps all evaluation metrics to empty lists
    group_field : str, optional
        The group field column, by default None
    user_field : str, optional
        The user field column, by default None
    learning_activity_field : str, optional
        The learning activity field column, by default None
    learning_activity_code_field : str, optional
        The learning activity code field column, by default None
    group : _type_, optional
        The group value, by default None
    user : _type_, optional
        The user value, by default None
    learning_activity : _type_, optional
        The learning_activity value, by default None
    learning_activity_code : _type_, optional
        The learning_activity_code value, by default None
    number_interactions_total : int, optional
        The number_interactions_total value, by default None
    number_attempts_total : int, optional
        The number_attempts_total value, by default None
    number_hints_total : int, optional
        The number_hints_total value, by default None
    time_total : float, optional
        The time_total value in seconds, by default None
    single_score : int, optional
        The single_score value, by default None
    single_score_hint_lowest : int, optional
        The single_score_hint_lowest value, by default None
    single_score_not_first_attempt_lowest : int, optional
        The single_score_not_first_attempt_lowest value, by default None
    score_highest : float, optional
        The score_highest value, by default None
    score_highest_without_hint : float, optional
        The score_highest_without_hint value, by default None
    score_first_attempt : float, optional
        The score_first_attempt value, by default None
    score_last_attempt : float, optional
        The score_last_attempt value, by default None
    number_interactions_until_score_highest : int, optional
        The number_interactions_until_score_highest value, by default None
    number_attempts_until_score_highest : int, optional
        The number_attempts_until_score_highest value, by default None
    number_hints_until_score_highest : int, optional
        The number_hints_until_score_highest value, by default None
    time_until_score_highest : float, optional
        The time_until_score_highest value in seconds, by default None
    is_correct : bool, optional
        The is_correct value, by default None
    is_correct_without_hint : bool, optional
        The is_correct_without_hint value, by default None
    is_correct_first_attempt : bool, optional
        The is_correct_first_attempt value, by default None
    is_correct_first_attempt_without_hint : bool, optional
        The is_correct_first_attempt_without_hint value, by default None
    is_correct_last_attempt : bool, optional
        The is_correct_last_attempt value, by default None
    is_correct_last_attempt_without_hint : bool, optional
        The is_correct_last_attempt_without_hint value, by default None
    number_interactions_until_correct : int, optional
        The number_interactions_until_correct value, by default None
    number_attempts_until_correct : int, optional
        The number_attempts_until_correct value, by default None
    number_hints_until_correct : int, optional
        The nubmer_hints_until_correct value, by default None
    time_until_correct : float, optional
        The time_until_correct value, by default None
    """
    if group_field:
        evaluation_learning_activity_merge_dict[group_field].append(group)

    evaluation_learning_activity_merge_dict[user_field].append(user)
    evaluation_learning_activity_merge_dict[learning_activity_field].append(
        learning_activity
    )
    evaluation_learning_activity_merge_dict[learning_activity_code_field].append(
        learning_activity_code
    )
    # learning activity evaluation fields
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_INTERACTIONS_TOTAL_FIELD_NAME_STR
    ].append(number_interactions_total)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_ATTEMPTS_TOTAL_FIELD_NAME_STR
    ].append(number_attempts_total)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_HINTS_TOTAL_FIELD_NAME_STR
    ].append(number_hints_total)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_TIME_IN_SEC_TOTAL_FIELD_NAME_STR
    ].append(time_total)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SINGLE_SCORE_FIELD_NAME_STR
    ].append(single_score)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SINGLE_SCORE_HINT_LOWEST_FIELD_NAME_STR
    ].append(single_score_hint_lowest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SINGLE_SCORE_NOT_FIRST_ATTEMPT_LOWEST_FIELD_NAME_STR
    ].append(single_score_not_first_attempt_lowest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SCORE_HIGHEST_FIELD_NAME_STR
    ].append(score_highest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SCORE_HIGHEST_WITHOUT_HINTS_FIELD_NAME_STR
    ].append(score_highest_without_hint)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SCORE_FIRST_ATTEMPT_FIELD_NAME_STR
    ].append(score_first_attempt)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_SCORE_LAST_ATTEMPT_FIELD_NAME_STR
    ].append(score_last_attempt)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_INTERACTIONS_UNTIL_HIGHEST_SCORE_FIELD_NAME_STR
    ].append(number_interactions_until_score_highest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_ATTEMPTS_UNTIL_HIGHEST_SCORE_FIELD_NAME_STR
    ].append(number_attempts_until_score_highest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_HINTS_UNTIL_HIGHEST_SCORE_FIELD_NAME_STR
    ].append(number_hints_until_score_highest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_TIME_IN_SEC_UNTIL_HIGHEST_SCORE_FIELD_NAME_STR
    ].append(time_until_score_highest)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIELD_NAME_STR
    ].append(is_correct)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_WITHOUT_HINTS_FIELD_NAME_STR
    ].append(is_correct_without_hint)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIRST_ATTEMPT_FIELD_NAME_STR
    ].append(is_correct_first_attempt)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_FIRST_ATTEMPT_WITHOUT_HINTS_FIELD_NAME_STR
    ].append(is_correct_first_attempt_without_hint)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_LAST_ATTEMPT_FIELD_NAME_STR
    ].append(is_correct_last_attempt)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_IS_CORRECT_LAST_ATTEMPT_WITHOUT_HINTS_FIELD_NAME_STR
    ].append(is_correct_last_attempt_without_hint)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_INTERACTIONS_UNTIL_CORRECT_FIELD_NAME_STR
    ].append(number_interactions_until_correct)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_ATTEMPTS_UNTIL_CORRECT_FIELD_NAME_STR
    ].append(number_attempts_until_correct)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_HINTS_UNTIL_CORRECT_FIELD_NAME_STR
    ].append(number_hints_until_correct)
    evaluation_learning_activity_merge_dict[
        EVALUATION_LEARNING_ACTIVITY_TIME_IN_SEC_UNTIL_CORRECT_FIELD_NAME_STR
    ].append(time_until_correct)


def add_evaluation_fields_learning_activity_decorator_with_arguments(
    interactions_df: pd.DataFrame,
    group_field: str,
    user_field: str,
    learning_activity_field: str,
    timestamp_field: str,
    order_field: str,
):
    """A decorator function which performs pre- and post-calculations for the eval_metric_func

    Parameters
    ----------
    interactions_df : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    timestamp_field : str
        The timestamp field column
    order_field : str
        The order field column

    """

    def decorator_add_evaluation_fields_learning_activity(eval_metric_func):
        def wrapper_add_evaluation_fields_learnig_activity(*args, **kwargs):
            # add learning activity identifier code field -> a sequence of uninterrupted learning activities with the same name will receive the same identifier code
            (
                interactions,
                learning_activity_code_field,
            ) = add_unique_identifier_code_to_learning_activity_in_sequence(
                interactions_df,
                group_field,
                user_field,
                learning_activity_field,
                timestamp_field,
                order_field,
            )
            # generate dictionary used for merging
            evaluation_learning_activity_merge_dict = (
                generate_evaluation_learning_activity_merge_dict(
                    group_field,
                    user_field,
                    learning_activity_field,
                    learning_activity_code_field,
                )
            )

            # use non-None values for grouping list
            grouping_list = [
                group_field,
                user_field,
                learning_activity_field,
                learning_activity_code_field,
            ]
            grouping_list = [i for i in grouping_list if i]
            has_group = group_field != None

            # loops over learning activity sequences(grouped into uninterrupted learning activities of the same name) and calculates evaluation metrics for these sequences
            for fields, df in interactions.groupby(grouping_list):
                if has_group:
                    group, user, learning_activity, learning_activity_code = fields
                else:
                    user, learning_activity, learning_activity_code = fields
                    group = None

                (
                    number_interactions_total,
                    number_attempts_total,
                    number_hints_total,
                    time_total,
                    single_score,
                    single_score_hint_lowest,
                    single_score_not_first_attempt_lowest,
                    score_highest,
                    score_highest_without_hint,
                    score_first_attempt,
                    score_last_attempt,
                    number_interactions_until_score_highest,
                    number_attempts_until_score_highest,
                    number_hints_until_score_highest,
                    time_until_score_highest,
                    is_correct,
                    is_correct_without_hint,
                    is_correct_first_attempt,
                    is_correct_first_attempt_without_hint,
                    is_correct_last_attempt,
                    is_correct_last_attempt_without_hint,
                    number_interactions_until_correct,
                    number_attempts_until_correct,
                    number_hints_until_correct,
                    time_until_correct,
                ) = eval_metric_func(df, **kwargs)

                # fills the merging dictionary with the calculated evaluation metrics at the learning activity level and identifier values
                fill_evaluation_learning_activity_merge_dict(
                    evaluation_learning_activity_merge_dict,
                    group_field=group_field,
                    user_field=user_field,
                    learning_activity_field=learning_activity_field,
                    learning_activity_code_field=learning_activity_code_field,
                    group=group,
                    user=user,
                    learning_activity=learning_activity,
                    learning_activity_code=learning_activity_code,
                    number_interactions_total=number_interactions_total,
                    number_attempts_total=number_attempts_total,
                    number_hints_total=number_hints_total,
                    time_total=time_total,
                    single_score=single_score,
                    single_score_hint_lowest=single_score_hint_lowest,
                    single_score_not_first_attempt_lowest=single_score_not_first_attempt_lowest,
                    score_highest=score_highest,
                    score_highest_without_hint=score_highest_without_hint,
                    score_first_attempt=score_first_attempt,
                    score_last_attempt=score_last_attempt,
                    number_interactions_until_score_highest=number_interactions_until_score_highest,
                    number_attempts_until_score_highest=number_attempts_until_score_highest,
                    number_hints_until_score_highest=number_hints_until_score_highest,
                    time_until_score_highest=time_until_score_highest,
                    is_correct=is_correct,
                    is_correct_without_hint=is_correct_without_hint,
                    is_correct_first_attempt=is_correct_first_attempt,
                    is_correct_first_attempt_without_hint=is_correct_first_attempt_without_hint,
                    is_correct_last_attempt=is_correct_last_attempt,
                    is_correct_last_attempt_without_hint=is_correct_last_attempt_without_hint,
                    number_interactions_until_correct=number_interactions_until_correct,
                    number_attempts_until_correct=number_attempts_until_correct,
                    number_hints_until_correct=number_hints_until_correct,
                    time_until_correct=time_until_correct,
                )

            # generate a pandas dataframe from the evaluation_learning_activity_merge_dict and transform booleans to integers
            evaluation_learning_activity_df = pd.DataFrame(
                evaluation_learning_activity_merge_dict
            )
            evaluation_learning_activity_df = evaluation_learning_activity_df.replace(
                {False: 0, True: 1}
            )

            # merge evaluation_learning_activity_merge_dict df with the interactions dataframe
            # -> for each learning activity a set of possible evaluation metrics is calculated (None if a particular metric is not available due to lack of evaluation data)
            merge_field_list = [
                group_field,
                user_field,
                learning_activity_field,
                learning_activity_code_field,
            ]
            merge_field_list = [i for i in merge_field_list if i]
            interactions = interactions.merge(
                evaluation_learning_activity_df, how="left", on=merge_field_list
            )

            return interactions

        return wrapper_add_evaluation_fields_learnig_activity

    return decorator_add_evaluation_fields_learning_activity


def add_evaluation_fields_learning_activity(
    calculate_eval_metrics_func,
    interactions_df: pd.DataFrame,
    group_field: str,
    user_field: str,
    learning_activity_field: str,
    timestamp_field: str,
    order_field: str,
    **kwargs,
):
    """Adds evaluation metrics on the learning activity level to the interaction dataframe and returns it.
    group_field and order_field can be set to None if the interactions dataframe does not contain those fields.

    Parameters
    ----------
    calculate_eval_metrics_func
        A function which takes the kwargs and calculates the evaluation metrics at the learning activity level
    interactions_df : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    timestamp_field : str
        The timestamp field column
    order_field : str
        The order field column

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added evaluation metrics at the learning activity level
    """
    interactions = add_evaluation_fields_learning_activity_decorator_with_arguments(
        interactions_df,
        group_field,
        user_field,
        learning_activity_field,
        timestamp_field,
        order_field,
    )(calculate_eval_metrics_func)(**kwargs)

    return interactions


def add_evaluation_fields_learning_activity_avg(
    interactions: pd.DataFrame,
    to_be_averaged_fields_list: list,
    averaged_fields_list: list,
    group_field: str,
    user_field: str,
    averaging_func,
):
    """Adds the average(the particular metric is specified via averaging_func - can be mean or median) of specified learning activity evaluation fields to the interactions dataframe.
    If group_field is not None the average will be calculated over group-user combinations(when calculating averages for groups), otherwise over
    users only(when calculating averages for the whole course).

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    to_be_averaged_fields_list : list
        A list of evaluation fields to be averaged
    averaged_fields_list : list
        A list of field names for the averaged evaluation metrics
    group_field : str
        The group field column
    user_field : str
        The user field column
    averaging_func : _type_
        A function which calculates an average over an input array

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added average evaluation fields
    """
    rename_dict = {
        k: v for k, v in zip(to_be_averaged_fields_list, averaged_fields_list)
    }

    group_list = [group_field, user_field]
    group_list = [i for i in group_list if i]
    evaluation_metrics_learning_activity = interactions.groupby(group_list)[
        to_be_averaged_fields_list
    ].apply(averaging_func, axis=0)
    evaluation_metrics_learning_activity = evaluation_metrics_learning_activity.rename(
        columns=rename_dict
    )
    evaluation_metrics_learning_activity = (
        evaluation_metrics_learning_activity.reset_index()
    )

    interactions = interactions.merge(
        evaluation_metrics_learning_activity, how="left", on=group_list
    )

    return interactions


def add_evaluation_fields_group_avg(
    interactions: pd.DataFrame,
    to_be_averaged_fields_list: list,
    averaged_fields_list: list,
    group_field: str,
    user_field: str,
    averaging_func,
):
    """Adds the average(the particular metric is specified via averaging_func - can be mean or median) of specified group evaluation fields to the interactions dataframe.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    to_be_averaged_fields_list : list
        A list of evaluation fields to be averaged
    averaged_fields_list : list
        A list of field names for the averaged evaluation metrics
    group_field : str
        The group field column
    user_field : str
        The user field column
    averaging_func : _type_
        A function which calculates an average over an input array
    """
    rename_dict = {
        k: v for k, v in zip(to_be_averaged_fields_list, averaged_fields_list)
    }
    evaluation_metrics_group = (
        interactions.groupby([user_field, group_field])
        .head(1)
        .groupby(user_field)[to_be_averaged_fields_list]
        .apply(averaging_func, axis=0)
    )
    evaluation_metrics_group = evaluation_metrics_group.rename(columns=rename_dict)
    evaluation_metrics_group = evaluation_metrics_group.reset_index()

    interactions = interactions.merge(
        evaluation_metrics_group, how="left", on=user_field
    )

    return interactions


def add_evaluation_correct_fields(
    interactions: pd.DataFrame,
    evaluation_group_score_field: str,
    evaluation_group_is_correct_field: str,
    evaluation_course_score_field: str,
    evaluation_course_is_correct_field: str,
    add_evaluation_group_is_correct_field: bool,
    add_evaluation_course_is_correct_field: bool,
    evaluation_group_score_correct_threshold: float,
    evaluation_course_score_correct_threshold: float,
    relational_operator_group,
    relational_operator_course,
):
    """Add is_correct fields to the interactions dataframe based on the relation of score values to the specified threshold level.
    If a score field is None, an is_correct field will not be added.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interaction dataframe
    evaluation_group_score_field : str
        The evaluation group score field
    evaluation_group_is_correct_field : str
        The the evaluation group is_correct field
    evaluation_course_score_field : str
        The evaluation course field
    evaluation_course_is_correct_field : str
        The the evaluation course is_correct field
    add_evaluation_group_is_correct_field : bool
        A flag indicating whether a group level is_correct field will be calculated
    add_evaluation_course_is_correct_field : bool
        A flag indicating whether a course level is_correct field will be calculated
    evaluation_group_score_correct_threshold : float
        The threshold for the group score values to be evaluated as correct
    evaluation_course_score_correct_threshold : float
        The threshold for the course score values to be evaluated as correct
    relational_operator_group :
        The relational operator used for the comparison between group score values and the group threshold
    relational_operator_course :
        The relational operator used for the comparison between course score values and the course threshold

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added is_correct fields
    """
    evaluation_is_correct_fields_list = [
        evaluation_group_is_correct_field,
        evaluation_course_is_correct_field,
    ]
    field_threshold_iterator = zip(
        [evaluation_group_score_field, evaluation_course_score_field],
        [
            evaluation_group_score_correct_threshold,
            evaluation_course_score_correct_threshold,
        ],
        [relational_operator_group, relational_operator_course],
        [
            EVALUATION_GROUP_IS_CORRECT_FIELD_NAME_STR,
            EVALUATION_COURSE_IS_CORRECT_FIELD_NAME_STR,
        ],
        [add_evaluation_group_is_correct_field, add_evaluation_course_is_correct_field],
    )

    for n, (
        score_field,
        threshold,
        rel_op,
        is_correct_field_name,
        add_is_correct_field,
    ) in enumerate(field_threshold_iterator):
        if add_is_correct_field:
            interactions[is_correct_field_name] = rel_op(
                interactions[score_field].apply(lambda x: int(x) if pd.notna(x) else x),
                threshold,
            )
            evaluation_is_correct_fields_list[n] = is_correct_field_name

    interactions = interactions.replace({False: 0, True: 1})

    return interactions, evaluation_is_correct_fields_list


def rename_and_add_empty_evaluation_fields_group_and_course(
    interactions: pd.DataFrame,
    evaluation_group_score_field: str,
    evaluation_group_is_correct_field: str,
    evaluation_course_score_field: str,
    evaluation_course_is_correct_field: str,
):
    """Rename the group/course evaluation fields. If these fields are not inferrable from the data, a None field will be created.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    evaluation_group_score_field : str
        The evaluation group score field
    evaluation_group_is_correct_field : str
        The evaluation group is_correct field
    evaluation_course_score_field : str
        The evaluation course score field
    evaluation_course_is_correct_field : str
        The evaluation course is_correct field

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with renamed or newly created evaluation fields
    """
    field_name_arguments_list = [
        evaluation_group_score_field,
        evaluation_group_is_correct_field,
        evaluation_course_score_field,
        evaluation_course_is_correct_field,
    ]
    evaluation_group_and_course_fields = (
        EVALUATION_GROUP_FIELD_LIST + EVALUATION_COURSE_FIELD_LIST
    )
    eval_field_name_mapping_list = list(
        zip(field_name_arguments_list, evaluation_group_and_course_fields)
    )
    rename_dict = {k: v for k, v in eval_field_name_mapping_list if k}
    interactions = interactions.rename(columns=rename_dict)

    for old, new in eval_field_name_mapping_list:
        if not old:
            interactions[new] = None

    return interactions


def add_evaluation_group_and_course_fields(
    interactions: pd.DataFrame,
    evaluation_group_score_field: str,
    evaluation_group_is_correct_field: str,
    evaluation_course_score_field: str,
    evaluation_course_is_correct_field: str,
    add_evaluation_group_is_correct_field: bool,
    add_evaluation_course_is_correct_field: bool,
    evaluation_group_score_correct_threshold: float,
    evaluation_course_score_correct_threshold: float,
    relational_operator_group,
    relational_operator_course,
):
    """Adds and renames evaluation fields on the group and course level.
    If score fields exist but not is_correct fields, the is_correct fields will be calculated using the specified threshold.
    Non inferrable group and course evaluation fields will be set to None.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    evaluation_group_score_field : str
        The evaluation group score field
    evaluation_group_is_correct_field : str
        The evaluation group is_correct field
    evaluation_course_score_field : str
        The evaluation course score field
    evaluation_course_is_correct_field : str
        The evaluation course is_correct field
    add_evaluation_group_is_correct_field : bool
        A flag indicating whether a group level is_correct field will be calculated
    add_evaluation_course_is_correct_field : bool
        A flag indicating whether a course level is_correct field will be calculated
    evaluation_group_score_correct_threshold : float
        The threshold for the group score values to be evaluated as correct
    evaluation_course_score_correct_threshold : float
        The threshold for the course score values to be evaluated as correct
    relational_operator_group :
        The relational operator used for the comparison between group score values and the group threshold
    relational_operator_course :
        The relational operator used for the comparison between course score values and the course threshold

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added and renamed evaluation group and course fields
    """
    interactions, evaluation_is_correct_fields_list = add_evaluation_correct_fields(
        interactions,
        evaluation_group_score_field,
        evaluation_group_is_correct_field,
        evaluation_course_score_field,
        evaluation_course_is_correct_field,
        add_evaluation_group_is_correct_field,
        add_evaluation_course_is_correct_field,
        evaluation_group_score_correct_threshold,
        evaluation_course_score_correct_threshold,
        relational_operator_group,
        relational_operator_course,
    )

    evaluation_group_is_correct_field = evaluation_is_correct_fields_list[0]
    evaluation_course_is_correct_field = evaluation_is_correct_fields_list[1]

    interactions = rename_and_add_empty_evaluation_fields_group_and_course(
        interactions,
        evaluation_group_score_field,
        evaluation_group_is_correct_field,
        evaluation_course_score_field,
        evaluation_course_is_correct_field,
    )

    return interactions


def add_evaluation_fields(
    interactions: pd.DataFrame,
    group_field: str | None,
    user_field: str,
    learning_activity_field: str,
    timestamp_field: str,
    order_field: str,
    evaluation_group_score_field: str,
    evaluation_group_is_correct_field: str,
    evaluation_course_score_field: str,
    evaluation_course_is_correct_field: str,
    add_evaluation_group_is_correct_field: bool,
    add_evaluation_course_is_correct_field: bool,
    evaluation_group_score_correct_threshold: float,
    evaluation_course_score_correct_threshold: float,
    relational_operator_group,
    relational_operator_course,
    calculate_eval_metrics_func,
    **kwargs,
) -> pd.DataFrame:
    """Adds evaluation metrics on the learning activity, group and course level to the interaction dataframe and returns it.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    user_field : str
        The user field column
    learning_activity_field : str
        The learning activity field column
    timestamp_field : str
        The timestamp field column
    order_field : str
        The order field column
        This argument should be set to None if the interactions dataframe does not have an order_field
    evaluation_group_score_field : str
        The evaluation group score field,
        This argument should be set to None if the interactions dataframe does not have an evaluation_group_score_field
    evaluation_group_is_correct_field : str
        The evaluation group is_correct field
        This argument should be set to None if the interactions dataframe does not have an evaluation_group_is_correct_field
    evaluation_course_score_field : str
        The evaluation course score field
        This argument should be set to None if the interactions dataframe does not have an evaluation_course_score_field
    evaluation_course_is_correct_field : str
        The evaluation course is_correct field
        This argument should be set to None if the interactions dataframe does not have an evaluation_course_is_correct_field
    add_evaluation_group_is_correct_field : bool
        A flag indicating whether a group level is_correct field will be calculated based on the specified evaluation_group_score_correct_threshold
    add_evaluation_course_is_correct_field : bool
        A flag indicating whether a course level is_correct field will be calculated based on the specified evaluation_course_score_correct_threshold
    evaluation_group_score_correct_threshold : float
        The threshold for the group score values to be evaluated as correct
    evaluation_course_score_correct_threshold : float
        The threshold for the course score values to be evaluated as correct
    relational_operator_group : _type_
        The relational operator used for the comparison between group score values and the group threshold (e.g. operator.gt(x,y))
    relational_operator_course : _type_
        The relational operator used for the comparison between course score values and the course threshold (e.g. operator.gt(x,y))
    calculate_eval_metrics_func : _type_
        A function which takes the kwargs and calculates and evaluation metrics at the learning activity level
    **kwargs
        Additional parameters passed into the calculate_eval_metrics_func

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added evaluation fields
    """
    # add evaluation fields per learning activity
    interactions = add_evaluation_fields_learning_activity(
        calculate_eval_metrics_func,
        interactions,
        group_field,
        user_field,
        learning_activity_field,
        timestamp_field,
        order_field,
        **kwargs,
    )

    # eliminate repeated learning activities -> information of repeated learning activity is extracted in method add_evaluation_fields_learning_activity
    interactions = keep_last_repeated_learning_activities(
        interactions, group_field, user_field, learning_activity_field, timestamp_field
    )

    # evaluation fields for group and course level
    interactions = add_evaluation_group_and_course_fields(
        interactions,
        evaluation_group_score_field,
        evaluation_group_is_correct_field,
        evaluation_course_score_field,
        evaluation_course_is_correct_field,
        add_evaluation_group_is_correct_field,
        add_evaluation_course_is_correct_field,
        evaluation_group_score_correct_threshold,
        evaluation_course_score_correct_threshold,
        relational_operator_group,
        relational_operator_course,
    )

    # replace all None values with np.na
    interactions = interactions.fillna(value=np.nan)

    return interactions


def add_evaluation_fields_average(
    interactions: pd.DataFrame, group_field: str | None, averaging_func
) -> pd.DataFrame:
    """Adds average evaluation metrics to the interaction dataframe and returns it. Learning activity evaluation metrics
    will be averaged over groups and the whole course. Group evaluation metrics are averaged over the whole course.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    group_field : str
        The group field column
        This argument should be set to None if the interactions dataframe does not have a group_field
    averaging_func : _type_
        A function which calculates an average over an input array (np.mean, np.median)
        Used for calculating the averages of learning activity/group evaluation metrics over groups and courses.

    Returns
    -------
    pd.DataFrame
        The interactions dataframe with added average evaluation fields
    """

    # group level learning activity average
    if group_field:
        interactions = add_evaluation_fields_learning_activity_avg(
            interactions,
            EVALUATION_LEARNING_ACTIVITY_FIELD_LIST,
            EVALUATION_GROUP_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST,
            GROUP_FIELD_NAME_STR,
            USER_FIELD_NAME_STR,
            averaging_func,
        )
    else:
        interactions[EVALUATION_GROUP_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST] = len(
            EVALUATION_GROUP_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST
        ) * [None]

    # course level learning activity average
    interactions = add_evaluation_fields_learning_activity_avg(
        interactions,
        EVALUATION_LEARNING_ACTIVITY_FIELD_LIST,
        EVALUATION_COURSE_ALL_LEARNING_ACTIVITIES_MEAN_FIELD_LIST,
        None,
        USER_FIELD_NAME_STR,
        averaging_func,
    )

    # course level group average
    if group_field:
        interactions = add_evaluation_fields_group_avg(
            interactions,
            EVALUATION_GROUP_FIELD_LIST,
            EVALUATION_COURSE_ALL_GROUPS_MEAN_FIELD_LIST,
            GROUP_FIELD_NAME_STR,
            USER_FIELD_NAME_STR,
            averaging_func,
        )
    else:
        interactions[EVALUATION_COURSE_ALL_GROUPS_MEAN_FIELD_LIST] = len(
            EVALUATION_COURSE_ALL_GROUPS_MEAN_FIELD_LIST
        ) * [None]

    # replace all None values with np.na
    interactions = interactions.fillna(value=np.nan)

    return interactions
