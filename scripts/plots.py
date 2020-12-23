#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plots:
    """
    Creates plots for text columns: word_dist_plot()
    and for categorical columns: count_plot()

    ----------
    Input: method(dataframe, first column to process, last column to process)
    ----------
    Output: plot with distinction fraudulent/legitimate.
    """

    def __init__(self, dataframe, start, end):

        self.dataframe = dataframe
        self.start = start
        self.end = end

    @staticmethod
    def word_dist_plot(dataframe: pd.core.frame.DataFrame,
                       start: int = 0,
                       end: int = -1
                       ) -> pd.core.frame.DataFrame or int or int:
        """
        Creates distribution plots for text columns.
        It will calculate the length of the documents
        within these columns and represent the population
        of fraudulent and legit job ads for those columns.

        NOTE: the y axis will be truncated at 800 to provide
        a better visual of the fraudulent ads population.

        ----------
        Input: word_dist_plot(dataframe[, start_of_column_slicing=0, end_of_column_slicing=-1]).
        ----------
        Output: one distribution plot per column.
        """
        if isinstance(dataframe, pd.core.frame.DataFrame):
            for column in dataframe.columns[start:end]:
                legit_desc_words = dataframe[dataframe['Fraudulent'] == 0][column].str.len(
                )
                fraud_desc_words = dataframe[dataframe['Fraudulent'] == 1][column].str.len(
                )
                sns.distplot(legit_desc_words,
                             kde=False,
                             label='Legitimate job posts',
                             color=sns.color_palette('RdYlGn')[4],
                             hist_kws={"alpha": 1})
                sns.distplot(fraud_desc_words,
                             kde=False,
                             label='Fraudulent job posts',
                             color=sns.color_palette('RdYlGn')[1],
                             hist_kws={"alpha": 1})
                plt.xlabel('Word count')
                plt.ylabel('Number of posts')
                plt.ylim(0, 800)
                plt.title(f"How many words have been used in the job's {column}?")
                plt.legend()
                plt.show()
        else:
            raise TypeError('arguments must be pandas.DataFrame, int, int')

    @staticmethod
    def count_plot(dataframe: pd.core.frame.DataFrame,
                   start: int = 0,
                   end: int = -1
                   ) -> pd.core.frame.DataFrame or int or int:
        """
        Creates a count plot per categorical columns.
        It differentiates fraudulent and legitimate posts populations.

        ----------
        Input: count_plot(dataframe[, start_of_column_slicing=0, end_of_column_slicing=-1])
        ----------
        Output: one count plot per column.
        """
        if isinstance(dataframe, pd.core.frame.DataFrame):
            for column in dataframe.columns[start:end]:
                sns.countplot(x=column, hue='Fraudulent',
                              palette='RdYlGn_r', data=dataframe, saturation=1)
                plt.title(f'Is there a type of {column} targeted by fraudsters?')
                plt.xticks(rotation=90)
                plt.ylim(0, 1000)
                plt.legend()
                plt.show()
        else:
            raise TypeError('arguments must be pandas.DataFrame, int, int')
