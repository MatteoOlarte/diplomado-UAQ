import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def apply_dark_theme():
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid")

    # Personalizar colores y estilos
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    plt.rcParams['axes.facecolor'] = '#2b2b2b'
    plt.rcParams['axes.edgecolor'] = '#444444'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#ffffff'
    plt.rcParams['xtick.color'] = '#ffffff'
    plt.rcParams['ytick.color'] = '#ffffff'
