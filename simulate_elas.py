#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

import sys
import os
import pandas as pd

from simulator.code.ELASS import ELASS
from simulator.code.read_input_files import read_sim_settings
import simulator.magic_values.elass_settings as es
from datetime import datetime
import simulator.code.read_input_files as rdr

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)

    sim_set = read_sim_settings(
        os.path.join(
            es.DIR_SIM_SETTINGS,
            '2024-08-26',
            'CurrentELAS_validation_fakedata_2_2.yml'
        )
    )

    # Read in simulation settings
    simulator = ELASS(
        sim_set=sim_set,
        verbose=True
    )

    ml = simulator.simulate_allocation(
        verbose=False
    )

    simulator.sim_results.save_outcomes_to_file(
        patients=simulator.patients,
        cens_date=sim_set.SIM_END_DATE,
        obligations=simulator.obligations
    )
