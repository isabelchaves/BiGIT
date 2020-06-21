import re
import unicodedata

import nltk
import pandas as pd


class PreProcessing:
    """
    Class with all the necessary functions to process and tokenize an
    expression or list of expressions.
    """

    def __init__(self, language: dict):
        self.language = language
        self.stemmer = nltk.SnowballStemmer(language['stemmer'])

    def _make_lowercase(self, col: pd.Series):
        return col.str.lower()

    def _remove_bad_chars(self, expression):
        """
        Method that removes the bad characters from a string.

        Args:
            expression: A string or a list of strings.

        Returns:
            A string or a list of strings with the last period removed from.
        """

        bad_chars = ['"', "'", '/', ',', '.', '(', ')', '—', '&', ';', '$', '%'
                                                                            '‘', '’', '!', '?', '«', '»', '-', '<', '>',
                     '+', '#', '|', ':', '_',
                     '°', 'ª', 'º', '*']

        if isinstance(expression, str):
            for char in bad_chars:
                expression = expression.replace(char, ' ')
        elif isinstance(expression, list):
            expression = [token.replace(char, '') for char in bad_chars
                          for token in expression]
        else:
            raise ValueError('expression must be a string or list. '
                             'type {} was passed'.format(type(expression)))

        return expression

    def _remove_accents(self, expression):
        """
        Method that removes accents from a string.

        Args:
            expression: A string or a list of strings.

        Returns:
            A string or a list of strings with accents removed.
        """
        if isinstance(expression, str):
            expression = ''.join(c for c
                                 in unicodedata.normalize('NFD', expression)
                                 if unicodedata.category(c) != 'Mn')
        elif isinstance(expression, list):
            expression = (
                [''.join(c for c in unicodedata.normalize('NFD', word)
                         if unicodedata.category(c) != 'Mn')
                 for word in expression])
        else:
            raise ValueError('expression must be a string or list')

        return expression

    @staticmethod
    def _apply_lowercase(sentence_tokens):
        """
        Method that applies lower case to each individual token in a given list
        of tokens.

        Args:
            sentence_tokens (:obj:`list` of :obj:`str`): A list of strings
            (tokens).

        Returns:
            A list of strings with all letters in small caps.
        """
        return [token.lower() for token in sentence_tokens]

    @staticmethod
    def _replace_from_sentence_tokens(sentence_tokens, to_replace, replacement):
        """
        Method that replaces a substring in a token for a given list of tokens.

        Args:
            sentence_tokens (:obj:`list` of :obj:`str`): A list of strings
            (tokens). to_replace (str): The substring meant to be replaced.
            replacement (str): The substring meant to replace with.

        Returns:
            A list of strings with replaced substrings.
        """
        return [token.replace(to_replace, replacement)
                for token in sentence_tokens]

    def _simple_split(self, sentence):

        sentence = re.sub('''[^a-z0-9A-Z\u00C0-\u00FF \-'.]''', '', sentence)

        return nltk.word_tokenize(
            sentence, language=self.language['nltk'])

    def tokenizer(self, sentence: str) -> str:
        sentence_tokens = self._remove_accents(sentence)
        sentence_tokens = self._remove_bad_chars(sentence_tokens)
        sentence_tokens = self._simple_split(sentence_tokens)
        sentence_tokens = self._replace_from_sentence_tokens(sentence_tokens=sentence_tokens,
                                                             to_replace='..', replacement='.')
        sentence_tokens = self._apply_lowercase(sentence_tokens=sentence_tokens)

        sentence_tokens = ' '.join(str(x) for x in sentence_tokens)  # to keep the sentence

        return sentence_tokens
