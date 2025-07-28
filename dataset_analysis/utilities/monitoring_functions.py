from .configs.general_config import *
from .constants.constants import *
from .standard_import import *

def return_object_size(dataset_name: str,
                       globals_dict: list,
                       result_tables: Type[Any]) -> pd.DataFrame:
    """Returns a dataframe which contains the size of objects in the current namespace in mb.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset
    object_list : list
        A list of all objects in the current namespace. Use python's builtin dir() function.
    result_tables : Type[Any]
        The ResultTables object
        

    Returns
    -------
    pd.DataFrame
        The object size dataframe
    """
    name_list = []
    size_in_mb_list = []
    id_seen = set()
    mega_factor = 10**6

    result_objects = [MONITORING_INTERACTIONS_STR, MONITORING_RESULT_TABLES_STR]
    object_list = result_objects + [i for i in globals_dict.keys() if i not in result_objects]

    for object_name in object_list:

        object_name_eval = globals_dict[object_name]
        object_id = id(object_name_eval)

        if object_id not in id_seen:
            try:
                if isinstance(object_name_eval, pd.DataFrame):
                    mega_byte = object_name_eval.memory_usage(deep=True).sum() / mega_factor
                elif isinstance(object_name_eval, np.ndarray):
                    mega_byte = asizeof.asizeof(object_name_eval) / mega_factor
                elif hasattr(object_name_eval, '__dict__'):
                    object_bytes = 0
                    for k,v in object_name_eval.__dict__.items():
                        v_name = k
                        v_id = id(v)
                        if v_id not in id_seen:
                            if isinstance(v, pd.DataFrame):
                                object_bytes += v.memory_usage(deep=True).sum()
                            if isinstance(v, np.ndarray):
                                object_bytes += asizeof.asizeof(v)
                            id_seen.add(v_id)
                        else:
                            if not ((v_name[0].isupper()) or (v_name[0] == '_')):
                                if v_name in MONITORING_RESULT_TABLES_LIST:
                                 print(DASH_STRING)
                                 print(f'Reference to already counted object in object: {object_name}')
                                 print(f'Already accounted for object: {v_name}')
                    mega_byte = object_bytes / mega_factor 
                else: 
                    mega_byte = asizeof.asizeof(object_name_eval) / mega_factor

                name_list.append(object_name)
                size_in_mb_list.append(mega_byte)
                id_seen.add(object_id)
            except:
                raise Exception(f'Size of object {object_name} could not be determined!')
        else:
            if not ((object_name[0].isupper()) or (object_name[0] == '_')):
                if object_name in MONITORING_RESULT_TABLES_LIST:
                    print(DASH_STRING)
                    print(f'Object already accounted for: {object_name}')

    object_size_df = pd.DataFrame({DATASET_NAME_FIELD_NAME_STR: dataset_name,
                                   MONITORING_OBJECT_NAME_FIELD_NAME_STR: name_list,
                                   MONITORING_MEGABYTE_FIELD_NAME_STR: size_in_mb_list})
    object_size_df = object_size_df.sort_values(by=MONITORING_MEGABYTE_FIELD_NAME_STR, 
                                                ascending=False).reset_index(drop=True)

    # add data to result tables
    object_size_df_filter = object_size_df[MONITORING_OBJECT_NAME_FIELD_NAME_STR].isin(MONITORING_RESULT_TABLES_LIST)
    object_size_df_filtered = object_size_df.loc[object_size_df_filter, :].reset_index(drop=True).copy()
    result_tables.object_size_df = object_size_df_filtered.copy()

    return object_size_df