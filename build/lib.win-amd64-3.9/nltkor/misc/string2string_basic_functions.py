"""
string2string code
src = https://github.com/stanfordnlp/string2string


MIT License

Copyright (c) 2023 Mirac Suzgun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""

from typing import List, Union

# Take the Cartesian product of two lists of strings (or lists of lists of strings)
def cartesian_product(
    lst1: Union[List[str], List[List[str]]],
    lst2: Union[List[str], List[List[str]]],
    boolList: bool = False,
    list_of_list_separator: str = " ## ",
) -> Union[List[str], List[List[str]]]:
    """
    This function returns the Cartesian product of two lists of strings (or lists of lists of strings).

    Arguments:
        lst1: The first list of strings (or lists of lists of strings).
        lst2: The second list of strings (or lists of lists of strings).
        boolList: A boolean flag indicating whether the output should be a list of strings (or lists of lists of strings) (default: False).

    Returns:
        The Cartesian product of the two lists of strings (or lists of lists of strings).
    """
    if lst1 == []:
        return lst2
    elif lst2 == []:
        return lst1
    return [
        s1 + ("" if not (boolList) else list_of_list_separator) + s2
        for s1 in lst1
        for s2 in lst2
    ]