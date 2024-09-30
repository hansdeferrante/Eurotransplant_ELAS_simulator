#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

Rules for allocation

@author: H.C. de Ferrante
"""

from math import isnan
from simulator.code.utils import zip_recursively_to_tuple

import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.elass_settings as es
import simulator.magic_values.magic_values_rules as mgr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.code import entities


DEFAULT_MELD_ETCOMP_THRESH = 30

# NSE rules for ET countries. Tuples of
# index in NSE_ID, initial 30-d equivalent,
# and 90-day upgrade, and max equivalent.
NSE_UPGRADES = [
    (1, 10, 10, 40),
    (2, 15, 10, 40)
]


# Center information
FL_CENTER_CODES = {
    'A': mgr.AUSTRIA,
    'B': mgr.BELGIUM,
    'C': mgr.CROATIA,
    'G': mgr.GERMANY,
    'H': mgr.HUNGARY,
    'L': mgr.BELGIUM,
    'N': mgr.NETHERLANDS,
    'S': mgr.SLOVENIA,
    'O': mgr.OTHER
}

DICT_CENTERS_TO_REGIONS = {
    'GHOTP': 'GND',
    'GBCTP': 'GNO',
    'GESTP': 'GNW',
    'GHGTP': 'GND',
    'GHBTP': 'GBW',
    'GTUTP': 'GBW',
    'GRBTP': 'GBY',
    'GFMTP': 'GMI',
    'GMZTP': 'GMI',
    'GJETP': 'GOS',
    'GLPTP': 'GOS',
    'GBOTP': 'GNW',
    'GMLTP': 'GBY',
    'GMNTP': 'GNW',
    'GKITP': 'GND',
    'GGOTP': 'GND',
    'GMHTP': 'GBY',
    'GKLTP': 'GNW',
    'GMBTP': 'GOS',
    'GNBTP': 'GBY',
    'GAKTP': 'GNW',
    'GROTP': 'GNO',
    'GWZTP': 'GBY',
    'GHSTP': 'GMI',
    'GKMTP': 'GNW',
    'GBDTP': 'GNO',
    'GHATP': 'GOS',
    'GNOOR': 'GNO',
    'GBWOR': 'GBW',
    'GOSOR': 'GOS',
    'GMIOR': 'GMI',
    'GNWOR': 'GNW',
    'GBYOR': 'GBY',
    'GNDOR': 'GND',
    'GFRTP': 'GBW',
    'GLUTP': 'GND',
    'GMATP': 'GBW',
    'GULTP': 'GBW',
    'GGITP': 'GMI',
    'GFDTP': 'GMI',
    'GKSTP': 'GMI',
    'GMRTP': 'GMI',
    'GBBTP': 'GNW',
    'GDUTP': 'GNW',
    'GDRTP': 'GOS',
    'GFDTP': 'GMI',
    'GMRTP': 'GMI'
}

# Blood group rules which are applied to donor/recipient matches
# (tables 4.1.1-4.1.3 in the FS.)
RECIPIENT_ELIGIBILITY_TABLES = {
    (cn.TAB2, 1): ([cn.TYPE_OFFER], ['Split']),
    (cn.TAB2, 2): ([cn.D_PED, cn.R_PED], [True, True]),
    (cn.TAB1, 1): ([cn.D_ALLOC_COUNTRY], [mgr.GERMANY])
}
DEFAULT_BG_TAB_COLLE = cn.TAB3

# Blood group compatibility tables. Tab1 is used for German donors,
# tab2 for split donors and pediatric donor/recipients, tab3 is
# used for non-German donors (see above). Each blood group table
# is a dictionary. Keys are tuples (to disambiguate), where the
# first value refers to the type of blood group rule that is to
# be applied. The values are tuples of column variables/values,
# which must be equal to belong to the group.
BG_COMPATIBILITY_TAB = {
    cn.TAB1: {
        (cn.BGC_TYPE1, 1): ([cn.PATIENT_IS_HU], [True]),
        # Full compatibility for Slovenia (missing from manual)
        (cn.BGC_FULL, 1): (
            [cn.RECIPIENT_COUNTRY, cn.ANY_OBL],
            [mgr.SLOVENIA, True]
            ),
        (cn.BGC_FULL, 2): ([cn.R_ACO], [True]),
        (cn.BGC_TYPE1, 2): ([cn.MELD_GE_ETTHRESH], [True])
    },
    cn.TAB2: {
    },
    cn.TAB3: {
        (cn.BGC_FULL, 1): ([cn.ALLOCATION_INT], [False]),
        (cn.BGC_FULL, 2): ([cn.R_ACO], [True]),
        (cn.BGC_FULL, 3): (
            [cn.RECIPIENT_COUNTRY, cn.ANY_OBL, cn.ALLOCATION_INT],
            [mgr.SLOVENIA, True, True]
            ),
        (cn.BGC_TYPE1, 1): ([cn.PATIENT_IS_HU], [True]),
        (cn.BGC_TYPE1, 2): ([cn.MELD_GE_ETTHRESH], [True])
    }
}

BG_COMPATIBILITY_DEFAULTS = {
    cn.TAB1: cn.BGC_TYPE2,
    cn.TAB2: cn.BGC_FULL,
    cn.TAB3: cn.BGC_TYPE2
}

# Dictionary of blood group compatibility rules. For each rule,
# the key refers to the donor blood group, and the values
# are eligible recipients.
BLOOD_GROUP_COMPATIBILITY_DICT = {
    cn.BGC_FULL: {  # Full compatibility
        cn.BG_A: set((cn.BG_A, cn.BG_AB)),
        cn.BG_B: set((cn.BG_B, cn.BG_AB)),
        cn.BG_AB: set((cn.BG_AB,)),
        cn.BG_O: set((cn.BG_A, cn.BG_B, cn.BG_AB, cn.BG_O))
    },
    cn.BGC_TYPE1: {  # ET compatible
        cn.BG_A: set((cn.BG_A, cn.BG_AB)),
        cn.BG_B: set((cn.BG_B, cn.BG_AB)),
        cn.BG_AB: set((cn.BG_AB, )),
        cn.BG_O: set((cn.BG_B, cn.BG_O))
    },
    cn.BGC_TYPE2: {  # TPG compatible
            cn.BG_A: set((cn.BG_A, cn.BG_AB)),
            cn.BG_B: set((cn.BG_B, cn.BG_AB)),
            cn.BG_AB: set((cn.BG_AB,)),
            cn.BG_O: set((cn.BG_O,))
        },
}

BLOOD_GROUP_INCOMPATIBILITY_DICT = {
    k: set(cg.BG_LEVELS).difference(v)
    for k, v in BLOOD_GROUP_COMPATIBILITY_DICT[cn.BGC_FULL].items()
}

# Blood group priority rules
BG_PRIORITY_RULES = {
    (1,): ([cn.MELD_GE_ETTHRESH, cn.BGC_TYPE1], [True, True]),
    (2,): ([cn.BGC_TYPE2], [True])
}
DEFAULT_BG_PRIORITY = 3


# Match rank tier
MAIN_RANK_TIERS = {
    cn.WLIV: {
        # HU first. For Germany, always rank E.
        (cn.TIER_A, 1): (
            [cn.PATIENT_IS_HU, cn.D_ALLOC_COUNTRY],
            [True, mgr.GERMANY]),
        # For non-German ped. donors ped. HU patients always E, internationally
        # always E, nationally only  if ET-compatible (otherwise D).
        (cn.TIER_A, 2): (
            [cn.PATIENT_IS_HU, cn.D_PED, cn.R_PED],
            [True, True, True]),
        (cn.TIER_A, 3): (
            [cn.PATIENT_IS_HU, cn.D_PED, cn.ALLOCATION_INT],
            [True, True, True]),
        (cn.TIER_A, 4): (
            [cn.PATIENT_IS_HU, cn.D_PED, cn.BGC_TYPE1],
            [True, True, True]),
        (cn.TIER_D, 1): ([cn.PATIENT_IS_HU, cn.D_PED], [True, True]),
        # For non-German HU adult donors, international is HU.
        (cn.TIER_A, 5): (
            [cn.PATIENT_IS_HU, cn.ALLOCATION_INT],
            [True, True]),
        (cn.TIER_A, 6): ([cn.PATIENT_IS_HU, cn.BGC_TYPE1], [True, True]),
        (cn.TIER_D, 2): ([cn.PATIENT_IS_HU, cn.D_PED], [True, True]),
        (cn.TIER_B, 1): ([cn.R_ACO], [True])
    },
    cn.SPLIT: {
        # Germany first
        (cn.TIER_D, 1): (
            [cn.PATIENT_IS_HU, cn.D_ALLOC_COUNTRY],
            [True, mgr.GERMANY]),
        (cn.TIER_E, 1): ([cn.R_ACO, cn.D_ALLOC_COUNTRY], [True, mgr.GERMANY]),
        (cn.TIER_F, 1): ([cn.D_ALLOC_COUNTRY], [mgr.GERMANY]),
        # Belgian non-residents go here.

        # Other ET countries
        (cn.TIER_A, 1): ([cn.PATIENT_IS_HU, cn.ALLOCATION_LOC], [True, True]),
        (cn.TIER_B, 1): ([cn.R_ACO, cn.ALLOCATION_LOC], [True, True]),
        (cn.TIER_C, 1): ([cn.ALLOCATION_LOC], [True]),
        (cn.TIER_D, 2): ([cn.PATIENT_IS_HU], [True]),
        (cn.TIER_E, 2): ([cn.R_ACO], [True]),
        (cn.TIER_F, 2): ([cn.R_ACO], [False])
    }
}

MAIN_TIER_REVERSAL_DICTS = {
    k: dict(zip(g, g[::-1])) for k, g in cg.MATCH_TIERS.items()
}


DEFAULT_RANK_TIER = {
    cn.WLIV: cn.TIER_D,
    cn.SPLIT: cn.TIER_F
}

# Heart-beating and non-heartbeating are the same in Austria.
FIRST_LAYER_TIERS = {
    cn.WLIV: {
        cn.TIER_A: {
            cntry: {(1, 1): ([cn.D_PED, cn.R_PED], [True, True])} for cntry in es.ET_COUNTRIES_OR_OTHER
        },
        cn.TIER_B: {
            cntry: {(1, 1): ([cn.D_PED, cn.R_PED], [True, True])} for cntry in es.ET_COUNTRIES_OR_OTHER
        },
        cn.TIER_D: {
            mgr.AUSTRIA: {
                # Pediatric donors first
                # Pediatric patients first.
                (24, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, True, True]),
                (23, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL], [True, True, True]),
                (22, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC, cn.BG_IDENTICAL], [True, True, True, True]),
                (21, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC], [True, True, True]),
                (20, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (19, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                # Obligations next
                (18, 1): ([cn.D_PED, cn.ANY_OBL, cn.R_LOWWEIGHT], [True, True, True]),
                (17, 1): ([cn.D_PED, cn.ANY_OBL], [True, True]),
                # Nationally to adults. First, locally.
                (16, 1): ([cn.D_PED, cn.ALLOCATION_LOC, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 1]),
                (15, 1): ([cn.D_PED, cn.ALLOCATION_LOC, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 2]),
                (14, 1): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 1]),
                (13, 1): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 2]),
                (12, 1): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 3]),
                # Next, nationally.
                (11, 1): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 1]),
                (10, 1): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 2]),
                (9, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 1]),
                (8, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 2]),
                (7, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 3]),
                # International pediatrics
                (6, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (5, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                # Internatinal adults
                (4, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (3, 1): ([cn.D_PED], [True]),
                # Adult donors second.
                (8, 2): ([cn.ANY_OBL], [True]),
                (7, 2): ([cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, 1]),
                (6, 2): ([cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, 2]),
                (5, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 1]),
                (4, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 2]),
                (3, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 3]),
                (2, 2): ([cn.ALLOCATION_INT], [True])
            },
            mgr.BELGIUM: {
                # --- Heartbeating donors (non-DCD) ---
                # HB pediatric donors first.
                (24, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, False, True, True]),
                (23, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ANY_OBL], [True, True, False, True]),
                (22, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, False, True, True]),
                (21, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ALLOCATION_NAT], [True, True, False, True]),
                # Obligations to adults
                # Identical blood groups only get priority for pediatric patient-donors.
                # (20, 1): ([cn.D_DCD, cn.D_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [False, True, True, True]),
                (19, 1): ([cn.D_PED, cn.D_DCD, cn.ANY_OBL], [False, True, True]),
                # Low-weight national adults first
                (18, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, False, True, True, 1]),
                (17, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, False, True, True, 2]),
                (16, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 1]),
                (15, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 2]),
                (14, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 3]),
                # International pediatrics
                (13, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.BG_IDENTICAL], [True, True, False, True]),
                (12, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD], [True, True, False]),
                # International adults
                (11, 1): ([cn.D_PED, cn.D_DCD, cn.R_LOWWEIGHT], [True, False, True]),
                (10, 1): ([cn.D_PED, cn.D_DCD], [True, False]),

                # HB adult donors
                (6, 1): ([cn.ANY_OBL, cn.D_DCD], [True, False]),
                (5, 1): ([cn.ALLOCATION_NAT, cn.D_DCD, cn.BG_PRIORITY], [True, False, 1]),
                (4, 1): ([cn.ALLOCATION_NAT, cn.D_DCD, cn.BG_PRIORITY], [True, False, 2]),
                (3, 1): ([cn.ALLOCATION_NAT, cn.D_DCD, cn.BG_PRIORITY], [True, False, 3]),
                (2, 1): ([cn.D_DCD], [False]),

                # --- Non-heartbeating donors (DCD) ---
                # NHB pediatric donors first.
                (28, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC, cn.BG_IDENTICAL], [True, True, True, True]),
                (27, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC], [True, True, True]),
                (26, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (25, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                # Local adult recipients second.
                (24, 2): ([cn.D_PED, cn.ALLOCATION_LOC, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 1]),
                (23, 2): ([cn.D_PED, cn.ALLOCATION_LOC, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 2]),
                (22, 2): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 1]),
                (21, 2): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 2]),
                (20, 2): ([cn.D_PED, cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, True, 3]),
                # National adult recipients second
                (19, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 1]),
                (18, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 2]),
                (17, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 1]),
                (16, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 2]),
                (15, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 3]),
                # International pediatrics
                (14, 2): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (13, 2): ([cn.D_PED, cn.R_PED], [True, True]),
                # International adults for pediatric donor.
                (11, 2): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (10, 2): ([cn.D_PED], [True]),
                # NHB adult donors
                (7, 2): ([cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, 1]),
                (6, 2): ([cn.ALLOCATION_LOC, cn.BG_PRIORITY], [True, 2]),
                (5, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 1]),
                (4, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 2]),
                (3, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 3]),
                (2, 2): ([cn.ALLOCATION_INT], [True]),
            },
            mgr.CROATIA: {
                # Pediatric donors first
                # Pediatric patients first.
                (20, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, True, True]),
                (19, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL], [True, True, True]),
                (18, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (17, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                # Obligations next
                (16, 1): ([cn.D_PED, cn.ANY_OBL, cn.R_LOWWEIGHT], [True, True, True]),
                (15, 1): ([cn.D_PED, cn.ANY_OBL], [True, True]),
                # Nationally to adults. First, locally.
                (13, 1): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 1]),
                (12, 1): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 2]),
                (11, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 1]),
                (10, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 2]),
                (9, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 3]),
                # International pediatrics
                (8, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (7, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                # Internatinal adults
                (6, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (5, 1): ([cn.D_PED], [True]),
                # Then, adult donors.
                (4, 1): ([cn.ANY_OBL], [True]),
                (3, 1): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 1]),
                (2, 1): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 2]),
                (1, 1): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 3]),
                (0, 1): ([cn.ALLOCATION_INT], [True])
            },
            mgr.GERMANY: {
                # Pediatric donors first
                # Pediatric patients first.
                (11, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, True, True]),
                (10, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL], [True, True, True]),
                (9, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (8, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                (7, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (6, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                # Obligations next
                (5, 1): ([cn.D_PED, cn.ANY_OBL], [True, True]),
                (4, 1): ([cn.D_PED, cn.ALLOCATION_NAT], [True, True]),
                (3, 1): ([cn.D_PED], [True]),
                # Adult donors second.
                (2, 1): ([cn.ANY_OBL], [True]),
                (1, 1): ([cn.ALLOCATION_NAT], [True]),
                (0, 1): ([cn.ALLOCATION_INT], [True])
            },
            mgr.HUNGARY: {
                # Pediatric donors first
                # Pediatric patients first.
                (24, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, True, True]),
                (23, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL], [True, True, True]),
                (22, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (21, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                # Obligations next
                (20, 1): ([cn.D_PED, cn.ANY_OBL, cn.R_LOWWEIGHT], [True, True, True]),
                (19, 1): ([cn.D_PED, cn.ANY_OBL], [True, True]),
                # Nationally to adults. First, locally.
                (18, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 1]),
                (17, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, True, True, 2]),
                (16, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 1]),
                (15, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 2]),
                (14, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 3]),
                # International pediatrics
                (13, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (12, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                # International adults
                (11, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (10, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                # Adult donors
                (6, 2): ([cn.ANY_OBL], [True]),
                (5, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 1]),
                (4, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 2]),
                (3, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 3]),
                (2, 2): ([cn.ALLOCATION_INT], [True])
            },
            mgr.NETHERLANDS: {
                # --- Heartbeating donors (non-DCD) ---
                # HB pediatric donors first.
                (24, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, False, True, True]),
                (23, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ANY_OBL], [True, True, False, True]),
                (22, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, False, True, True]),
                (21, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.ALLOCATION_NAT], [True, True, False, True]),
                # Obligations to adults
                (20, 1): ([cn.D_PED, cn.D_DCD, cn.ANY_OBL, cn.BG_IDENTICAL], [True, False, True, True]),
                (19, 1): ([cn.D_PED, cn.D_DCD, cn.ANY_OBL], [True, False, True]),
                # Low-weight national adults first
                (18, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, False, True, True, 1]),
                (17, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT, cn.BG_PRIORITY], [True, False, True, True, 2]),
                (16, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 1]),
                (15, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 2]),
                (14, 1): ([cn.D_PED, cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, False, True, 3]),
                # International pediatrics
                (13, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD, cn.BG_IDENTICAL], [True, True, False, True]),
                (12, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD], [True, True, False]),
                # International adults
                (11, 1): ([cn.D_PED, cn.D_DCD, cn.R_LOWWEIGHT], [True, False, True]),
                (10, 1): ([cn.D_PED, cn.D_DCD], [True, False]),

                # HB adult donors
                (6, 1): ([cn.D_DCD, cn.ANY_OBL], [False, True]),
                (5, 1): ([cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [False, True, 1]),
                (4, 1): ([cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [False, True, 2]),
                (3, 1): ([cn.D_DCD, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [False, True, 3]),
                (2, 1): ([cn.D_DCD], [False]),

                # --- Non-heartbeating donors (DCD) ---
                # Pediatric patients first
                (19, 2): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (18, 2): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                # National adult patients second.
                (17, 2): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 1]),
                (16, 2): ([cn.D_PED, cn.R_LOWWEIGHT, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, True, 2]),
                (15, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 1]),
                (14, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 2]),
                (13, 2): ([cn.D_PED, cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, True, 3]),
                # International pediatrics
                (12, 2): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (11, 2): ([cn.D_PED, cn.R_PED], [True, True]),
                # International adults for pediatric donor.
                (10, 2): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (9, 2): ([cn.D_PED], [True]),
                # NHB adult donors
                (5, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 1]),
                (4, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 2]),
                (3, 2): ([cn.ALLOCATION_NAT, cn.BG_PRIORITY], [True, 3]),
                (2, 2): ([cn.ALLOCATION_INT], [True]),
            },
            mgr.SLOVENIA: {
                # Pediatric donors first
                # First Slovenian national pediatrics, then obligations.
                (16, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT, cn.BG_IDENTICAL], [True, True, True, True]),
                (15, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                (14, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.BG_IDENTICAL], [True, True, True, True]),
                (13, 1): ([cn.D_PED, cn.R_PED, cn.ANY_OBL], [True, True, True]),
                # To local adults next
                (11, 1): ([cn.D_PED, cn.ALLOCATION_NAT, cn.R_LOWWEIGHT], [True, True, True]),
                (10, 1): ([cn.D_PED, cn.ALLOCATION_NAT], [True, True]),
                # Next to international pediatrics
                (8, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (7, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                # To adults internationally.
                (6, 1): ([cn.D_PED, cn.ANY_OBL, cn.R_LOWWEIGHT], [True, True, True]),
                (5, 1): ([cn.D_PED, cn.ANY_OBL], [True, True]),
                (4, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (3, 1): ([cn.D_PED], [True]),
                (2, 1): ([cn.ALLOCATION_NAT], [True]),
                (1, 1): ([cn.ANY_OBL], [True]),
                (0, 1): ([cn.ALLOCATION_INT], [True])
            },
            'Other': {
                # Pediatric donors first
                # First Slovenian national pediatrics, then obligations.
                (9, 1): ([cn.D_PED, cn.R_PED, cn.BG_IDENTICAL], [True, True, True]),
                (8, 1): ([cn.D_PED, cn.R_PED], [True, True]),
                (6, 1): ([cn.D_PED, cn.R_LOWWEIGHT], [True, True]),
                (5, 1): ([cn.D_PED], [True]),
                # To local adults next
                (2, 1): ([cn.D_PED], [False])
            }
        }
    },
    cn.SPLIT: {
        **dict.fromkeys(
            [cn.TIER_A, cn.TIER_B, cn.TIER_D, cn.TIER_E],
            {
                cntry: {
                    (1, 1): ([cn.D_PED, cn.R_PED], [True, True])
                } for cntry in es.ET_COUNTRIES_OR_OTHER
            }
        ),
        **dict.fromkeys(
            [cn.TIER_C, cn.TIER_F],
            {
                cntry: (
                    {
                        # ET split rules
                        (5, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC], [True, True, True]),
                        (4, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                        (3, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_INT], [True, True, True]),
                        (2, 1): ([cn.ALLOCATION_LOC], [True]),
                        (1, 1): ([cn.ALLOCATION_NAT], [True]),
                        (0, 1): ([cn.ALLOCATION_INT], [True])
                    }
                    if cntry != mgr.GERMANY else
                    # German split rules
                    {
                        (8, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, True, True]),
                        (6, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_INT], [True, True, True]),
                        (4, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_NAT], [True, False, True]),
                        (3, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_INT], [True, False, True]),
                        (2, 1): ([cn.ALLOCATION_NAT], [True]),
                        (1, 1): ([cn.ALLOCATION_INT], [True])
                    }
                ) for cntry in set(es.ET_COUNTRIES_OR_OTHER)
            }
        )
    }
}

MTCH_TIER_ORDERING = {
    'cols': [cn.MTCH_TIER, cn.MTCH_LAYER, cn.MTCH_OBL, cn.MTCH_LAYER_MELD, cn.MTCH_LAYER_REG, cn.MTCH_LAYER_WT],
    'asc': [False, False, True, False, False, False]
}


# Check whether obligation tier is applicable for a patient.
OBL_TIERS = [cn.TIER_D, cn.TIER_E]
OBL_TIER_APPLICABLE = {
    **dict.fromkeys(
        [mgr.SLOVENIA, mgr.HUNGARY, mgr.CROATIA, mgr.GERMANY, mgr.AUSTRIA],
        {
            (True, tier): ([cn.MTCH_TIER, cn.TYPE_OFFER], [tier, cn.WLIV])
            for tier in OBL_TIERS
        }
    ),
    **dict.fromkeys(
        [mgr.NETHERLANDS, mgr.BELGIUM],
        {
            (True, tier): ([cn.D_DCD, cn.MTCH_TIER, cn.TYPE_OFFER], [False, tier, cn.WLIV])
            for tier in OBL_TIERS
        }
    ),
    **dict.fromkeys(
        ['Other'],
        {}
    )
}

# --- Settings for when center offers may be made ---- #
CENTER_OFFER_GROUPS = {
    'CZATP': 'CHROR', 'CZMTP': 'CHROR', 'CRITP': 'CHROR', 'CZPTP': 'CHROR',
    'HBSTP': 'HUNOR'
}

# Tiers in which non-local center offers can be done.
CENTER_OFFER_TIERS = {
    cn.WLIV: {
        (True, 1): ([cn.MTCH_TIER], [cn.TIER_C]),
        (True, 2): ([cn.MTCH_TIER], [cn.TIER_D])
    },
    cn.SPLIT: {
        (True, 1): ([cn.MTCH_TIER], [cn.TIER_F])
    }
}

RULES_LOCAL_CENTER_OFFER = {
    cn.LOCAL_PEDIATRIC_CENTER_OFFER: {
        **dict.fromkeys(
            [mgr.SLOVENIA, mgr.HUNGARY, mgr.CROATIA],
            {(True, 1): ([cn.D_PED, cn.R_PED], [True, True])}
        ),
        mgr.AUSTRIA: {(True, 1): ([cn.D_PED, cn.R_PED, cn.ALLOCATION_LOC], [True, True, True])},
        mgr.BELGIUM: {(True, 1): ([cn.D_PED, cn.R_PED, cn.D_DCD], [True, True, True])},
        mgr.GERMANY: {(True, 1): ([cn.D_PED, cn.R_PED, cn.TYPE_OFFER_DETAILED], [True, True, 31])}
    },
    cn.LOCAL_GENERAL_CENTER_OFFER: {
        **dict.fromkeys(
            [mgr.SLOVENIA, mgr.HUNGARY, mgr.CROATIA],
            {
                (True, 2): ([cn.R_PED], [False]),
                (True, 1): ([cn.R_PED], [True])
            }
        ),
        mgr.AUSTRIA: {(True, 1): ([cn.ALLOCATION_LOC], [True])},
        mgr.BELGIUM: {(True, 1): ([cn.ALLOCATION_LOC, cn.D_DCD], [True, True])},
        mgr.GERMANY: {(True, 1): ([cn.TYPE_OFFER_DETAILED], [31])}
    }
}


RULES_INT_CENTER_OFFER = {
    cn.PEDIATRIC_CENTER_OFFER_INT: {
            **{
                (True, cntry): (
                    [cn.D_PED, cn.R_PED, cn.ALLOCATION_INT, cn.ANY_OBL, cn.RECIPIENT_COUNTRY],
                    [True, True, True, True, cntry])
                for cntry in [mgr.SLOVENIA, mgr.HUNGARY, mgr.CROATIA]
            },
            (True, mgr.AUSTRIA): ([cn.D_PED, cn.R_PED, cn.ANY_OBL, cn.RECIPIENT_COUNTRY], [True, True, True, mgr.AUSTRIA]),
            (True, mgr.BELGIUM): ([cn.D_PED, cn.ALLOCATION_LOC, cn.R_PED, cn.D_DCD, cn.RECIPIENT_COUNTRY], [True, True, True, False, mgr.BELGIUM])
    },
    cn.GENERAL_CENTER_OFFER_INT: {
        **{
            (True, cntry): ([cn.ANY_OBL, cn.ALLOCATION_INT, cn.RECIPIENT_COUNTRY], [True, True, cntry])
            for cntry in [mgr.SLOVENIA, mgr.HUNGARY, mgr.CROATIA]
        },
        (True, mgr.AUSTRIA): ([cn.ANY_OBL, cn.RECIPIENT_COUNTRY], [True, mgr.AUSTRIA]),
        (True, mgr.BELGIUM): ([cn.D_DCD, cn.RECIPIENT_COUNTRY], [True, mgr.BELGIUM])
    }
}


# Fourth match rank tiers (i.e. priority for regional in Germany)
FOURTH_MRL = [
    (
        [
            cn.TYPE_OFFER, cn.MTCH_TIER, cn.ALLOCATION_REG, cn.D_ALLOC_COUNTRY,
            ],
        [cn.SPLIT, cn.TIER_F, True, mgr.GERMANY]
    ),
    (
        [
            cn.MTCH_TIER, cn.ALLOCATION_REG, cn.D_ALLOC_COUNTRY, cn.TYPE_OFFER
            ],
        [cn.TIER_D, True, mgr.GERMANY, cn.WLIV]
    ),
    (
        [
            cn.MTCH_TIER, cn.ALLOCATION_REG,
            cn.D_ALLOC_COUNTRY, cn.TYPE_OFFER,
            ],
        [cn.TIER_E, True, mgr.GERMANY, cn.WLIV]
    ),
    (
        [
            cn.TYPE_OFFER, cn.MTCH_TIER,
            cn.ALLOCATION_LOC, cn.D_ALLOC_COUNTRY
            ],
        [cn.SPLIT, cn.TIER_F, True, mgr.GERMANY]
    ),
    (
        [
            cn.MTCH_TIER, cn.ALLOCATION_LOC,
            cn.D_ALLOC_COUNTRY, cn.TYPE_OFFER
            ],
        [cn.TIER_D, True, mgr.GERMANY, cn.WLIV]
    ),
    (
        [
            cn.MTCH_TIER, cn.ALLOCATION_LOC,
            cn.D_ALLOC_COUNTRY, cn.TYPE_OFFER
            ],
        [cn.TIER_E, True, mgr.GERMANY, cn.WLIV]
    )
]


# Functions to check whether donor/patient are pediatric
def check_ELAS_ped_don(donor: 'entities.Donor') -> bool:
    """Checks whether a donor is pediatric (i.e. <46kg)"""
    if not isnan(donor.__dict__[cn.D_WEIGHT]):
        return donor.__dict__[cn.D_WEIGHT] < 46
    return False


def check_ELAS_ped_rec(patient: 'entities.Patient', match_age: float) -> bool:
    """Checks whether a patient is pediatric (i.e. <16 years)"""
    if not isnan(match_age):
        return match_age <= 16
    return False


def check_rec_low_weight(patient: 'entities.Patient') -> bool:
    """Checks whether a donor is low-weight (i.e. <55kg)"""
    #assert r_weight is not None and not isnan(r_weight), \
    #    f'{r_weight} is not a valid weight.'
    if not isnan(patient.__dict__[cn.R_WEIGHT]):
        return patient.__dict__[cn.R_WEIGHT] < 55
    return False

# Zip rules
FIRST_LAYER_TIERS = zip_recursively_to_tuple(FIRST_LAYER_TIERS)
RECIPIENT_ELIGIBILITY_TABLES = zip_recursively_to_tuple(RECIPIENT_ELIGIBILITY_TABLES)
BG_COMPATIBILITY_TAB = zip_recursively_to_tuple(BG_COMPATIBILITY_TAB)
BG_PRIORITY_RULES = zip_recursively_to_tuple(BG_PRIORITY_RULES)
RULES_LOCAL_CENTER_OFFER = zip_recursively_to_tuple(RULES_LOCAL_CENTER_OFFER)
RULES_INT_CENTER_OFFER = zip_recursively_to_tuple(RULES_INT_CENTER_OFFER)
MAIN_RANK_TIERS = zip_recursively_to_tuple(MAIN_RANK_TIERS)
OBL_TIER_APPLICABLE = zip_recursively_to_tuple(OBL_TIER_APPLICABLE)
FOURTH_MRL = [list(zip(c[0], c[1])) for c in FOURTH_MRL]
CENTER_OFFER_TIERS = zip_recursively_to_tuple(CENTER_OFFER_TIERS)
