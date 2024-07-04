from .standard_import import *

def check_value_in_iterable(value: Any, 
                            iterable: Iterable) -> bool:
    if value not in iterable:
        raise ValueError(f'Iterable does not contain value: {value}!')
    return True

def check_value_not_none(value: Any,
                         error_message: str | None = None) -> bool:
    # check for object identity -> None is singleton
    if value is None:
        message_str = ''
        if error_message:
            message_str = '\n' + error_message 
        raise ValueError(f'Value was None!' + f'{message_str}')
    return True

def return_value_if_not_none(value: Any,
                             return_value_not_none: Any,
                             return_value_none: Any,
                             raise_error_if_none: bool,
                             error_message: str | None = None) -> Any:
    # check for object identity -> None is singleton
    if value is None:
        if raise_error_if_none:
            message_str = ''
            if error_message:
                message_str = '\n' + error_message 
            raise ValueError(f'Value was None!' + f'{message_str}')
        else:
            return return_value_none

    return return_value_not_none

def check_if_is_instance(object_: Any,
                         class_: Callable):
    if not isinstance(object_, class_):
        raise TypeError(f'Object is not of type {type(class_)}')
    
    return True


