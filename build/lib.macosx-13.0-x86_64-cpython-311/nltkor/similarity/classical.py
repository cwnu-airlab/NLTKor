"""
string2string similarity
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

"""
This module contains the classes for the similarity metrics and functions.
"""


from typing import List, Union, Tuple, Optional
import numpy as np

# # Import the LongestCommonSubsequence class

# for dev purposes
import sys
# sys.path.append("/Users/dowon/nltk_ko/nltk/metrics")
from nltkor.alignment import LongestCommonSubsequence, LongestCommonSubstring
# from alignment import LongestCommonSubsequence, LongestCommonSubstring

# Longest Common Subsequence based similarity class
class LCSubsequenceSimilarity(LongestCommonSubsequence):
    """
    This class contains the Longest Common Subsequence similarity metric.

    This class inherits from the LongestCommonSubsequence class.
    """

    def __init__(self):
        super().__init__()


    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        denominator: str = 'max',
    ) -> float:
        """
        Returns the LCS-similarity between two strings.

        Arguments:
            str1 (Union[str, List[str]]): The first string or list of strings.
            str2 (Union[str, List[str]]): The second string or list of strings.
            denominator (str): The denominator to use. Options are 'max' and 'sum'. Default is 'max'.

        Returns:
            float: The similarity between the two strings.

        Raises:
            ValueError: If the denominator is invalid.
        """

        # Get the numerator
        numerator, _ = super().compute(str1, str2)

        if denominator == 'max':
            return (numerator / max(len(str1), len(str2)))
        elif denominator == 'sum':
            return (2. * numerator / (len(str1) + len(str2)))
        else:
            raise ValueError('Invalid denominator.')



# Longest Common Substring based similarity class
class LCSubstringSimilarity(LongestCommonSubstring):
    """
    This class contains the Longest Common Substring similarity metric.

    This class inherits from the LongestCommonSubstring class.
    """
    def __init__(self):
        super().__init__()


    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        denominator: str = 'max',
    ) -> float:
        """
        Returns the LCS-similarity between two strings.

        Arguments:
            str1 (Union[str, List[str]]): The first string or list of strings.
            str2 (Union[str, List[str]]): The second string or list of strings.
            denominator (str): The denominator to use. Options are 'max' and 'sum'. Default is 'max'.

        Returns:
            float: The similarity between the two strings.

        Raises:
            ValueError: If the denominator is invalid.
        """
        # Get the numerator
        numerator, _ = super().compute(str1, str2)

        if denominator == 'max':
            return (numerator / max(len(str1), len(str2)))
        elif denominator == 'sum':
            return (2. * numerator / (len(str1) + len(str2)))
        else:
            raise ValueError('Invalid denominator.')


# Jaro similarity class
class JaroSimilarity:
    """
    This class contains the Jaro similarity metric.
    """

    def __init__(self):
        pass


    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> float:
        """
        This function returns the Jaro similarity between two strings.

        Arguments:
            str1 (Union[str, List[str]]): The first string or list of strings.
            str2 (Union[str, List[str]]): The second string or list of strings.

        Returns:
            float: The Jaro similarity between the two strings.
        """
        # Get the length of the strings
        len1 = len(str1)
        len2 = len(str2)

        # Get the maximum distance, which we denote by k
        k = max(len1, len2) // 2 - 1

        # Initialize the number of matching characters and the number of transpositions
        num_matches = 0
        num_transpositions = 0

        # Initialize the list of matching flags for the strings
        matches1 = [False] * len1
        matches2 = [False] * len2

        # Loop through the characters in the first string and find the matching characters
        for i in range(len1):
            # Get the lower and upper bounds for the search
            lower_bound = max(0, i - k)
            upper_bound = min(len2, i + k + 1)

            # Loop through the characters in the second string
            for j in range(lower_bound, upper_bound):
                # Check if the characters match
                if not matches2[j] and str1[i] == str2[j]:
                    # Increment the number of matches
                    num_matches += 1

                    # Set the matching flags
                    matches1[i] = True
                    matches2[j] = True

                    # Break out of the loop
                    break

        # Check if there are no matches
        if num_matches == 0:
            return 0.

        # Loop through again but this time find the number of transpositions
        # That is, the number of times where there are two matching characters but there is another "matched" character in between them
        moving_index = 0
        for i in range(len1):
            # Check if the character is a match
            if matches1[i]:
                # Find the next match
                for j in range(moving_index, len2):
                    # Check if the character is a match
                    if matches2[j]:
                        # Set the moving index
                        moving_index = j + 1

                        # Check if the characters are not in the right order
                        if str1[i] != str2[j]:
                            # Increment the number of transpositions
                            num_transpositions += 1

                        # Break out of the loop
                        break

        num_transpositions = num_transpositions // 2

        # Return the Jaro similarity
        return (num_matches / len1 + num_matches / len2 + (num_matches - num_transpositions) / num_matches) / 3.0

def demo():
    """
    This function demonstrates the similarity metrics.
    """
    # Initialize the similarity metrics
    lcs_sim = LCSubsequenceSimilarity()
    lcs_sub_sim = LCSubstringSimilarity()
    jaro_sim = JaroSimilarity()

    # Initialize the strings
    str1 = '제가 나와 있는 곳은 경남 거제시 옥포동 덕포 해수욕장에 나와 있습니다.'
    str2 = '강한 바람에 간판이나 지붕이 떨어지는 등 피해가 잇따르기도 했습니다.'

    # Get the similarity metrics
    lcs_sim_score = lcs_sim.compute(str1, str2)
    lcs_sub_sim_score = lcs_sub_sim.compute(str1, str2)
    jaro_sim_score = jaro_sim.compute(str1, str2)

    # Print the results
    print('Longest Common Subsequence Similarity: {}'.format(lcs_sim_score))
    print('Longest Common Substring Similarity: {}'.format(lcs_sub_sim_score))
    print('Jaro Similarity: {}'.format(jaro_sim_score))

if __name__ == '__main__':
    demo()
