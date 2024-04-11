from .standard_import import *
from .constants import *

def style_html_table_header(soup: Type[BeautifulSoup]) -> None:
    head_elements = soup.select('thead th')
    for head in head_elements:
        head['style'] = RESULT_TABLES_HEADER_STYLE

def style_html_table_index(soup: Type[BeautifulSoup]) -> None:
    index_elements = soup.select('tbody th')
    for index in index_elements:
        index['style'] = RESULT_TABLES_INDEX_STYLE

def style_html_table_values(soup: Type[BeautifulSoup]) -> None:
    value_elements = soup.select('tbody td')
    for value in value_elements:
        value['style'] = RESULT_TABLES_VALUE_STYLE

def style_html_table_true_values(soup: Type[BeautifulSoup]) -> None:
    true_elements = soup.find_all(lambda tag: tag.text.strip() == 'True')
    for true in true_elements:
        true['style'] = RESULT_TABLES_TRUE_STYLE

def style_html_table_false_values(soup: Type[BeautifulSoup]) -> None:
    false_elements = soup.find_all(lambda tag: tag.text.strip() == 'False')
    for false in false_elements:
        false['style'] = RESULT_TABLES_FALSE_STYLE

def style_html_table_dash_values(soup: Type[BeautifulSoup]) -> None:
    false_elements = soup.find_all(lambda tag: tag.text.strip() == '-')
    for false in false_elements:
        false['style'] = RESULT_TABLES_DASH_STYLE

def remove_empty_html_table_row(soup: Type[BeautifulSoup]) -> None:
    tr_elements = soup.find_all('tr')
    for tr in tr_elements:
        all_empty = all(value.text.strip() == '' for value in tr)
        if all_empty:
            tr.extract() 

def remove_empty_html_table_column(soup: Type[BeautifulSoup]) -> None:
    tr_elements = soup.find_all('tr')
    for tr in tr_elements:
        th_elements = tr.find_all('th', rowspan=None) 
        for th in th_elements: 
            if not th.text.strip():
                th.extract()
                break

def replace_html_table_na_value(soup: Type[BeautifulSoup]) -> None:
    na_values_list = [RESULT_TABLES_NAN_VALUE, RESULT_TABLES_NONE_VALUE, RESULT_TABLES_NA_VALUE, f'{np.nan}']
    all_tags = soup.find_all()
    for tag in all_tags:
        if tag.string in na_values_list:
            tag.string.replace_with(RESULT_TABLES_DASH_VALUE)

def replace_html_table_text(soup: Type[BeautifulSoup],
                            old_text: str,
                            new_text: str) -> None:
    all_tags = soup.find_all()
    for tag in all_tags:
        if tag.string == old_text:
            tag.string.replace_with(new_text)

def apply_html_table_formatting(soup: Type[BeautifulSoup]):
    """Apply standardized formatting and styling to an html table via BeautifulSoup

    Parameters
    ----------
    soup : Type[BeautifulSoup]
        A BeautifulSoup object
    """    
    # header
    style_html_table_header(soup)

    # index
    style_html_table_index(soup)

    # values
    style_html_table_values(soup)

    # booleans
    style_html_table_true_values(soup)
    style_html_table_false_values(soup)

    # replace NaNs with dashes
    replace_html_table_na_value(soup)
    style_html_table_dash_values(soup)
