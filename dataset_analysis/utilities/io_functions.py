from .standard_import import *
from .constants import *
from .config import *

def save_interaction_and_mapping_df(interactions: pd.DataFrame,
                                    field_mapping_dataframe: pd.DataFrame,
                                    value_mapping_dataframe: pd.DataFrame,
                                    path_to_dataset_folder: str,
                                    dataset_name: str):
    """Saves the interactions and fields_mapping dataframes. Transforms headers to snake case.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions dataframe
    fields_mapping_dataframe : pd.DataFrame
        The fields mapping dataframe
    path_to_dataset_folder : str
        The directory in which the the datasets are being saved
    dataset_name : str
        The name used for saving the interactions dataset
    """
    # interactions dataframe
    field_name_list = interactions.columns
    snake_case_field_name_list = [fn.replace(' ', '_').lower() for fn in field_name_list]
    interactions = interactions.reset_index(drop=True)
    interactions.to_csv(path_to_dataset_folder + dataset_name + '.csv', index=False, header=snake_case_field_name_list)

    # field mapping dataframe
    field_mapping_dataframe[NEW_FIELDNAME_FIELD_NAME_STR] = field_mapping_dataframe[NEW_FIELDNAME_FIELD_NAME_STR].map(lambda x: x.replace(' ', '_').lower() if x else x)
    field_name_list = field_mapping_dataframe.columns
    snake_case_field_name_list = [fn.replace(' ', '_').lower() for fn in field_name_list]
    field_mapping_dataframe.to_csv(path_to_dataset_folder + dataset_name + FIELD_MAPPING_DATAFRAME_NAME_STR + '.csv', index=False, header=snake_case_field_name_list)

    # value mapping dataframe
    field_name_list = value_mapping_dataframe.columns
    snake_case_field_name_list = [fn.replace(' ', '_').lower() for fn in field_name_list]
    value_mapping_dataframe.to_csv(path_to_dataset_folder + dataset_name + VALUE_MAPPING_DATAFRAME_NAME_STR + '.csv', index=False, header=snake_case_field_name_list)

def delete_all_pickle_files_within_directory(path_within_pickle_directory_list: list[str]) -> None:
    """Removes all pickle files within the specified directory 

    Parameters
    ----------
    path_within_pickle_directory_list : str
        A list of path elements pointing to a subfolder of the pickle directory indicating where the serialized object is being saved 
    """        
    path_to_directory = os.path.join(PATH_TO_PICKLED_OBJECTS_FOLDER, *path_within_pickle_directory_list)
    path_to_file = os.path.join(path_to_directory, '*.pickle')

    # first delete all existing pickle to prevent keeping files which are not being overwritten
    for f in glob.glob(path_to_file):
        os.remove(f)

def pickle_write(object_to_pickle,
                 path_within_pickle_directory_list: list[str],
                 filename: str,
                 delete_old_files=False) -> None:
    """Serializes a python object and stores it in the specified location

    Parameters
    ----------
    object_to_pickle : 
        A python object to be serialized
    path_within_pickle_directory_list : list[str]
        A list of path elements pointing to a subfolder of the pickle directory indicating where the serialized object is being saved 
    filename : str
        The name given to the serialized python object (without the pickle file extension)
    delete_old_files : bool
        A flag indicating whether all existing pickle files in path_within_pickle_directory should be deleted
    """
    filename = filename + '.pickle'
    path_to_directory = os.path.join(PATH_TO_PICKLED_OBJECTS_FOLDER, *path_within_pickle_directory_list)
    path_to_file = os.path.join(path_to_directory, filename)
    
    if delete_old_files:
        delete_all_pickle_files_within_directory(path_within_pickle_directory_list)

    with open(path_to_file, 'wb') as f:
        pickle.dump(object_to_pickle, f, pickle.HIGHEST_PROTOCOL)

def pickle_read(path_within_pickle_directory_list: list[str],
                filename) -> Any:
    """Reads and returns a serialized python object located in the specified directory 

    Parameters
    ----------
    path_within_pickle_directory_list : str
        A list of path elements pointing to a subfolder of the pickle directory indicating where the serialized object is being saved 
    filename : str
        The name given to the serialized python object

    Returns
    -------
        The deserialized python object
    """        
    filename = filename + '.pickle'
    path_to_directory = os.path.join(PATH_TO_PICKLED_OBJECTS_FOLDER, *path_within_pickle_directory_list)
    path_to_file = os.path.join(path_to_directory, filename)

    with open(path_to_file, 'rb') as f:
        pickled_object = pickle.load(f)

    return pickled_object

def return_pickled_files_list(path_within_pickle_directory_list: list[str],
                              *args) -> list[str]:
    """Returns a list of filenames (without .pickle) of all serialized python object located in the specified directory 

    Parameters
    ----------
    path_within_pickle_directory_list : str
        A list of path elements pointing to a subfolder of the pickle directory indicating where the serialized objects are being located 

    Returns
    -------
        A list of filenames
    """        
    path_to_directory = os.path.join(PATH_TO_PICKLED_OBJECTS_FOLDER, *path_within_pickle_directory_list)
    path_to_file = os.path.join(path_to_directory, '*.pickle')

    split_function = lambda x: x.split('/')[-1].split('.')[0]
    sort_function = lambda x: int(x.split('_')[-1])

    pickle_files_list = glob.glob(path_to_file)
    if args:
        for arg in args:
            pickle_files_list = [file_str for file_str in pickle_files_list if arg in file_str]
    pickle_files_list = map(split_function, pickle_files_list)
    pickle_files_list = sorted(pickle_files_list, key=sort_function)

    return pickle_files_list

def return_pickle_path_list_and_name(dataset_name: str,
                                     pickle_subdirectory: str,
                                     pickle_name: str) -> tuple:
    """Returns the pathlist and name for a pickle file

    Parameters
    ----------
    dataset_name : str
        The name of the dataset
    pickle_subdirectory : str
        The subdirectory where the pickle file will be saved
    pickle_name : str
        The name of the pickle file

    Returns
    -------
    tuple
        A tuple consisting of the pathlist and name of a pickle file 
    """        

    path_list = [pickle_subdirectory, dataset_name]
    name = dataset_name + pickle_name

    return path_list, name