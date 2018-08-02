#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
from skfeature.function.information_theoretical_based import MRMR


expro = pd.read_csv(sys.argv[1],index_col=False)

X, y = np.array(expro.ix[:, 1:]), np.array(expro.ix[:, 0])

idx = MRMR.mrmr(X, y, n_selected_features=500)
markers = expro.ix[:,idx]
markers.to_csv(sys.argv[2],index=False)

