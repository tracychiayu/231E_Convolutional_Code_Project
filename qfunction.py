from scipy.special import erfc
from decimal import Decimal, getcontext
import numpy as np

getcontext().prec = 20

def Q(x):
    return 0.5 * erfc(float(x) / float(np.sqrt(2)))