"""
Central configuration for the Drift-Sense localization pipeline.

Every tunable used by driftsense/ and evaluation/ lives here, so there is
exactly one place to change a threshold rather than hunting through modules.
Values are the ones empirically validated in DriftSense_System_Documentation.md
(30/30 success @5px, 0.0px mean error on the real dram_octagonal dataset).
"""

import numpy as np

# --- Geometry -----------------------------------------------------------
# The generators embed a 100x100 patch (downsampled 10x from a 1000x1000
# reference) into a 1000x1000 search canvas. Per the hackathon spec, the
# embedded patch also gets rotation (1-3 deg) and scale jitter (-20% to
# +20%) applied before embedding -- so both sweeps below must cover that
# range, not just the nominal 0.1 ratio.
SEARCH_SIZE = 1000
EMBED_SIZE = 100
# widened from (0.085, 0.115) to +-20% around the nominal 0.1 ratio, to
# match the generators' scale jitter range (see fixed_noise_data_*.py)
DEFAULT_SCALES = np.linspace(0.08, 0.12, 9)
# rotation search, degrees -- matches the generators' 1-3 deg polygon rotation
DEFAULT_ANGLES = np.linspace(-3.0, 3.0, 7)
# --- Matching -------------------------------------------------------------
NMS_RADIUS_PX = 20.0        # merge candidates within this radius across scales
TIE_SCORE_TOLERANCE = 0.03  # candidates within this much of the top score are "tied"
MIN_SCORE = 0.35            # below this, matched=False (validated: genuine 0.55-0.76, false 0.13)

# --- Evaluation -----------------------------------------------------------
SUCCESS_TOLERANCES_PX = (1, 3, 5)  # report success rate at each of these tolerances

# --- Preprocessing ----------------------------------------------------------
# CLAHE is OFF by default: the validated 100%/0px result was achieved WITHOUT
# it. Turn on only if a future, harder-noise style needs the contrast boost --
# and re-run evaluate.py to confirm it doesn't regress accuracy before trusting it.
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# --- Styles -----------------------------------------------------------------
STYLES = ["dram_octagonal", "dram_6f2", "finfet_sram", "beol_interconnect"]
DATA_DIR_FOR_STYLE = {
    "dram_octagonal": "data_1",
    "dram_6f2": "data_2",
    "finfet_sram": "data_3",
    "beol_interconnect": "data_4",
}