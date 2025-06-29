"""Top-level package for MOREL."""

__author__ = """Sedjro Salomon Hotegni"""
__email__ = 'salomon.hotegni@tu-dortmund.de'
__version__ = '0.1.3'

from .advermorel import MOREL
from .losses import morel_loss, mart_loss,  trades_loss, loat_loss
from .models import morelnet
