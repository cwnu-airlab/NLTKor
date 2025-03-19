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

"""
This file contains the default tokenizer.
"""

from typing import List

# Tokenizer class
class Tokenizer:
    """
    This class contains the tokenizer.
    """

    def __init__(self,
        word_delimiter: str = " ",
    ):
        """
        Initializes the Tokenizer class.

        Arguments:
            word_delimiter (str): The word delimiter. Default is " ".
        """
        # Set the word delimiter
        self.word_delimiter = word_delimiter

    # Tokenize
    def tokenize(self,
        text: str,
    ) -> List[str]:
        """
        Returns the tokens from a string.

        Arguments:
            text (str): The text to tokenize.

        Returns:
            List[str]: The tokens.
        """
        return text.split(self.word_delimiter)
    
    # Detokenize
    def detokenize(self,
        tokens: List[str],
    ) -> str:
        """
        Returns the string from a list of tokens.

        Arguments:
            tokens (List[str]): The tokens.

        Returns:
            str: The string.
        """
        return self.word_delimiter.join(tokens)