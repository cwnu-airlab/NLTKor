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
    This module contains the hash functions used in search algorithms.

    A hash function takes a string (or other object) and returns a number. 
    The number is called the hash value, hash code, or simply the hash. The hash value is used to determine the location of the string in the hash table.
    - The hash function must be deterministic, meaning that the same string always produces the same hash value. 
    - If two strings produce the same hash value, we say that the hash values collide. 
    - The hash function must also be fast, so it is important to keep the number of operations to a minimum.
"""

from typing import List, Union, Tuple, Optional
import numpy as np


# A parent class for all hash functions
class HashFunction:
    """
    This class contains the parent class for all hash functions.
    """
    def __init__(self):
        pass

    def compute(self,
        str1: str,
    ) -> int:
        """
        Returns the hash value of a string.

        Arguments:
            str1 (str): The string.

        Returns:
            int: The hash value of the string.
        """
        pass


# Polynomial rolling hash function class
class PolynomialRollingHash(HashFunction):
    """
    This class contains the polynomial rolling hash function.
    """

    def __init__(self,
        base: int = 10, # 256,
        modulus: int = 101, # 65537,
    ) -> None:
        """
        Initializes the polynomial rolling hash function.

        Arguments:
            base (int): The base to use. Default is 256.
            modulus (int): The modulus to use. Default is 65537.

        Returns:
            None

        .. note::
            * Why 65537? Because it is a Fermat prime.
        """
        super().__init__()

        # Check the inputs
        assert base > 0, 'The base must be positive.'
        assert modulus > 0, 'The modulus must be positive.'
        
        # Set the attributes
        self.base = base
        self.modulus = modulus

        # Initialize the current hash value
        self.current_hash = 0


    def compute(self,
        str1: str,
    ) -> int:
        """
        Returns the hash value of a string.

        Arguments:
            str1 (str): The string.

        Returns:
            int: The hash value of the string.
        """
        # Compute the hash value of the string
        for char in str1:
            self.current_hash = (self.current_hash * self.base + ord(char)) % self.modulus

        # Return the hash value
        return self.current_hash
    

    def update(self,
        old_char: str,
        new_char: str,
        window_size: int,
    ) -> int:
        """
        Updates the hash value of a string.

        Arguments:
            old_char (str): The old character.
            new_char (str): The new character.

        Returns:
            int: The hash value of the string.
        """
        # Update the hash value of the string
        self.current_hash = (self.current_hash - ord(old_char) * (self.base ** (window_size - 1))) % self.modulus
        self.current_hash = (self.current_hash * self.base + ord(new_char)) % self.modulus

        # Return the hash value
        return self.current_hash
    

    def reset(self) -> None:
        """
        Resets the hash value.

        Arguments:
            None

        Returns:
            None
        """
        # Reset the current hash value
        self.current_hash = 0