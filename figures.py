"""A program for making figures for each year."""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

import seaborn as sns


def inputs(matrix, vector):
    """"read in csv files for matrices and vectors, returns them as dict."""
    mat = np.array(pd.read_csv(matrix).ix[:, 2:])
    vec = pd.read_csv(vector)
    behavior = np.array(vec.ix[:, -1:])
    countries = np.array(vec.ix[:, 1:2])
    values = {'matrix': mat, 'behavior': behavior, 'country names': countries}
    return values


def mdscoordinates(matrix):
    """calculate the metric MDS coordinates."""
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed",
                       random_state=2)
    coordinates = mds.fit_transform(matrix)
    return coordinates


def mdsfigure(coordinates, year, behavior, countries):
    """draw a nice metric MDS plot for the year."""
    fig = plt.figure(figsize=(28.0, 16.5))
    fig.suptitle('Revolutionary Situations in ' + str(year),
                 fontsize=18, fontweight='bold')
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', c=behavior)

    for label, x, y, in zip(countries, coordinates[:, 0], coordinates[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round, pad=0.5',
                               fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))
    leg = plt.legend(('Revolutions', 'Stable'), loc='lower right',
                     frameon=True)
    leg.get_frame().set_edgecolor('b')
    fig.savefig('images/'+str(year)+'.png', dpi=100)
