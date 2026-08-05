from .ufr import UFR
from .hyperuof import HyperUOF
from .dpr import DPR
from .fairdual import FairDual
from .cpfair import CPFair
from .multifr import MultiFR
from .ada2fair import Ada2Fair
from .fair_method import FAIRMethod
from .fairsort import FairSort
from .popularity_ips import PopularityIPS


BASELINES = {
    'ufr': UFR,
    'hyperuof': HyperUOF,
    'dpr': DPR,
    'fairdual': FairDual,
    'cpfair': CPFair,
    'multifr': MultiFR,
    'ada2fair': Ada2Fair,
    'fair': FAIRMethod,
    'fairsort': FairSort,
    'popularity_ips': PopularityIPS,
}


CATEGORIES = {
    'ufr': 'user-side',
    'hyperuof': 'user-side',
    'dpr': 'item-side',
    'fairdual': 'item-side',
    'cpfair': 'two-sided',
    'multifr': 'two-sided',
    'ada2fair': 'two-sided',
    'fair': 'two-sided',
    'fairsort': 'two-sided',
    'popularity_ips': 'item-side',
}


PROCESSING_TYPES = {
    'ufr': 'post-processing',
    'hyperuof': 'in-processing',
    'dpr': 'in-processing',
    'fairdual': 'in-processing',
    'cpfair': 'post-processing',
    'multifr': 'in-processing',
    'ada2fair': 'in-processing',
    'fair': 'in-processing',
    'fairsort': 'post-processing',
    'popularity_ips': 'in-processing',
}
