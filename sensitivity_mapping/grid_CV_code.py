# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:56:07 2018

@author: Greta
"""

# Utility function to move the midpoint of a colormap to be around
# the values of interest.
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
    
def plotHeatmapParam(inputarray,title='Heatmap'):
    reshape_scores = np.reshape(inputarray,[10,10]) #Change CV fold no
    
    # Inputfile for C and gamma values
    os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\decoding\\CV_2out')
    data = xlrd.open_workbook('CV_val+test_1_2.xlsx').sheet_by_index(0)
    sub1, sub2, C_vals, gamma_vals, train_scores, test_scores1, test_scores2 = getValsCV2out(data) # Min funktion til at tage scores og info fra .csv
        
    random_chance = 0.565217391304347
    C_range = C_vals[0:10]
    gamma_range = gamma_vals[0::10]

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(reshape_scores, interpolation='nearest', cmap=plt.cm.hot, 
               norm=MidpointNormalize(vmin=random_chance - 0.04, midpoint=random_chance),vmax=random_chance+0.04)
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.colorbar()
    plt.yticks(np.arange(len(gamma_range)), gamma_range)
    plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
    plt.title(title)
    
    plt.show()