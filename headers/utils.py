'''
Author: Ludovic Carlu
'''

import numpy as np
import pandas as pd
import scipy.stats


def boostrap_oob(df_input):
    """
    :param df_input: Df which contains as columns features and y
    :return boostrap_df and and oob_df (oob=out of bag)
    """
    bootstrap = df_input.sample(len(df_input.index), replace=True)
    oob_index = [x for x in df_input.index if x not in bootstrap.index]
    oob = df_input.iloc[oob_index]

    return bootstrap, oob
