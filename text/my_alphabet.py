from enum import Enum


class BasicAlphabet:
    """
    Class represents structure of known signs
    """
    SIGNS = {"F", "T", "!", "@", "#", "$", "&", ",", ".", " ",
             "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
             "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
             "u", "v", "w", "x", "y", "z", "\n", ":", ";", "?",
             "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "+", "-", "*", "/", "^", "%", "=", "_", "[", "]",
             "{", "}", "`", "~", "<", ">", "'", "|", "\"", "", "E"}

    def get_list(self):
        return list(self.SIGNS)

