import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

rcPath='/Users/yasmene/rc/'

################################### hard coded ###################################
colorDic = {
    'coulomb': '#9A9A9A',

    # base model blocks
    'dft':  '#9E3F3F',
    'mm':   '#E0B72F',
    'gb':   '#2B6FA3',

    # correction arrows
    'mmx':  "#E9894E",
    'gbx':  '#2CA6A4',
    'gbxx': "#9B6BE8",
}

qYlimDic = {-2: (-5, 10),
            -1: (-5, 5),
             0: (-5, 5),
             1: (-5, 5),
             2: (-5, 5)}

xLim = (0.2, 1.3)
fillOpacity = 0.5

################################### graphing helpers ###################################

def loadPMF(file):
    if os.path.isfile(file):
        data = np.loadtxt(file)
        print(f'Loaded: {file}')
        return data
    else:
        print(f'NOT FOUND: {file}')  
        return file

def setZero(c,G,c0, G0=0):
    zero_ndx=min(range(len(c)), key=lambda i: abs(c[i]-c0)) #np.where(c==c0)[0][0]
    G_offset=[G[i]-G[zero_ndx]+G0 for i in range(len(G))]
    return G_offset

def interpR(data, rVals):
    from scipy.interpolate import interp1d
    # Create interpolation function for base data
    f = interp1d(data[:,0], data[:,1], kind='linear', fill_value="extrapolate")
    # r values for interpolation
    rMin = min(data[0,0], rVals[0])
    rMax = max(data[-1,0], rVals[-1])
    # Create new data array with interpolated values
    interpData = np.array([
        [r, f(r)] if rMin <= r <= rMax else [r, 0]
        for r in rVals
    ])
    return interpData

def getY(data, xVal):
    # Find the index of the closest x value
    idx = (np.abs(data[:,0] - xVal)).argmin()
    return data[idx,1]

def coulomb(r0, rmax, qprod, eps=78.5):
    rList = np.linspace(r0, rmax, 100)
    e = 1.6e-19
    nm = 1e-9
    A = 6.022e23
    k_kCal = 1/(4 * np.pi * 8.854187817e-12) / 4.184e3
    const = k_kCal * e**2 / nm * A
    return rList, const * qprod / (eps * rList)

def coulombZero(rList, qprod, rZero, eps=78.5):
    e = 1.6e-19
    nm = 1e-9
    A = 6.022e23
    k_kCal = 1/(4 * np.pi * 8.854187817e-12) / 4.184e3
    const = k_kCal * e**2 / nm * A
    V = const * qprod / (eps * rList)
    V_offset = setZero(rList, V, rZero, G0=0)
    return np.array([rList, V_offset]).T

################################### Graphs ###################################

from matplotlib.collections import PathCollection

def pubPMF(pairInfo, label=None, show_ylabel=True):
    ax = plt.gca()
    qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

    ax.set_xlim(xLim)

    ylims = qYlimDic.get(qprod)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel(r"$r$ (nm)", fontsize=9)

    if show_ylabel:
        ax.set_ylabel(r"PMF (kcal mol$^{-1}$)", fontsize=9)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    if label is not None:
        ax.set_title(label, fontsize=9, pad=3)

    ax.tick_params(
        axis="both", which="major",
        direction="in", length=4, width=0.8,
        labelsize=8, top=True, right=True
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", length=2, width=0.6,
        top=True, right=True
    )
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    for line in ax.lines:
        line.set_linewidth(1.4)

    # Only resize scatter markers, NOT fill_between collections
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            coll.set_sizes([30])

    # remove legends in publication plots
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# def pubPMF(pairInfo, label=None, show_ylabel=True):
#     ax = plt.gca()
#     qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

#     ax.set_xlim(xLim)
#     ylims = qYlimDic.get(qprod)
#     if ylims is not None:
#         ax.set_ylim(ylims)

#     ax.set_xlabel(r"$r$ (nm)", fontsize=9)

#     if show_ylabel:
#         ax.set_ylabel(r"PMF (kcal mol$^{-1}$)", fontsize=9)
#     else:
#         ax.set_ylabel("")
#         ax.tick_params(labelleft=False)

#     if label is not None:
#         ax.set_title(label, fontsize=9, pad=3)

#     ax.tick_params(
#         axis="both", which="major",
#         direction="in", length=4, width=0.8,
#         labelsize=8, top=True, right=True
#     )
#     ax.tick_params(
#         axis="both", which="minor",
#         direction="in", length=2, width=0.6,
#         top=True, right=True
#     )
#     ax.minorticks_on()

#     for spine in ax.spines.values():
#         spine.set_linewidth(0.8)

#     for line in ax.lines:
#         line.set_linewidth(1.4)

#     for coll in ax.collections:
#         if hasattr(coll, "set_sizes"):
#             coll.set_sizes([30])

#     # force-remove legends in publication plots
#     leg = ax.get_legend()
#     if leg is not None:
#         leg.remove()

def graphGBX(pairInfo, publication=False):
    sysTitle = pairInfo['A']['name'] + " -- " + pairInfo['B']['name']
    qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

    # Check that data is available
    corrData = pairInfo.get('gbx', {}).get('diffData')
    if corrData is None: print("No GB* data available for this pair.") ; return

    # Load base data
    baseData = pairInfo['gb']['pmfData']
    # Load transition radii
    mm_rMax = pairInfo.get('gbx', {}).get('mm_rMax')
    mm_rMax = (mm_rMax, getY(pairInfo['mm']['pmfData'], mm_rMax))
    # load reference data
    ref1Data = pairInfo['mm']['pmfData']
    coulombData = coulombZero(baseData[:,0], qprod, mm_rMax[0])
    # trim reference data to end at corresponding rMax
    ref1Data = ref1Data[ref1Data[:,0] <= mm_rMax[0]]
    # Align GB* correction with base data
    corrData = pairInfo['gbx']['diffData']
    corrNewR = interpR(corrData, baseData[:,0])
    basePlusCorr = baseData[:,1] + corrNewR[:,1]

    # Plot GB* correction
    plt.fill_between(corrNewR[:,0], baseData[:,1], 
                     basePlusCorr, color=colorDic['gbx'], 
                     alpha=fillOpacity, label='GB* Correction') 
    # Plot reference data
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='MM PMF', color=colorDic['mm'])
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--')
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='GB PMF', color=colorDic['gb'])
    # Plot transition radii
    plt.scatter(mm_rMax[0], mm_rMax[1], color=colorDic['mm'], marker='o', s=100, label='rMax_MM')
    
    # Plot options
    if publication:
        pubPMF(pairInfo, label="GB*")
    else:
        plt.title(sysTitle + " GB*")
        plt.xlabel("Distance (nm)")
        plt.ylabel("PMF (kcal/mol)")
        plt.ylim(qYlimDic.get(qprod))
        plt.xlim(xLim)
        plt.legend()

    
def graphGBXX(pairInfo, publication=False):
    sysTitle = pairInfo['A']['name'] + " -- " + pairInfo['B']['name']
    qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

    # Check that data is available
    corrData = pairInfo.get('gbxx', {}).get('diffData')
    if corrData is None: print("No GB** data available for this pair.") ; return

    # Load base data
    baseData = pairInfo['gb']['pmfData']
    # Load transition radii
    dft_rMax = pairInfo.get('gbxx', {}).get('dft_rMax')
    dft_rMax = (dft_rMax, getY(pairInfo['dft']['pmfData'], dft_rMax))
    mm_rMin = pairInfo.get('gbxx', {}).get('mm_rMin')
    mm_rMin = (mm_rMin, getY(pairInfo['mm']['pmfData'], mm_rMin))
    mm_rMax = pairInfo.get('gbxx', {}).get('mm_rMax')
    mm_rMax = (mm_rMax, getY(pairInfo['mm']['pmfData'], mm_rMax))
    # load reference data
    ref1Data = pairInfo['mm']['pmfData']
    ref2Data = pairInfo['dft']['pmfData']
    coulombData = coulombZero(baseData[:,0], qprod, mm_rMax[0])
    # trim reference data to end at corresponding rMax
    ref1Data = ref1Data[ref1Data[:,0] <= mm_rMax[0]]
    ref2Data = ref2Data[ref2Data[:,0] <= dft_rMax[0]]
    # Align GB* correction with base data
    corrData = pairInfo['gbxx']['diffData']
    corrNewR = interpR(corrData, baseData[:,0])
    basePlusCorr = baseData[:,1] + corrNewR[:,1]

    # Plot GB** correction
    plt.fill_between(corrNewR[:,0], baseData[:,1], 
                     basePlusCorr, color=colorDic['gbxx'], 
                     alpha=fillOpacity, label='GB** Correction')
    # Plot reference data
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='MM PMF', color=colorDic['mm'])
    plt.plot(ref2Data[:,0], ref2Data[:,1], label='DFT PMF', color=colorDic['dft'])
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--')
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='GB PMF', color=colorDic['gb'])
    # Plot transition radii
    plt.scatter(dft_rMax[0], dft_rMax[1], color=colorDic['dft'], marker='o', s=100, label='rMax_DFT')
    plt.scatter(mm_rMin[0], mm_rMin[1], color=colorDic['mm'], marker='o', s=100, label='rMin_MM')
    plt.scatter(mm_rMax[0], mm_rMax[1], color=colorDic['mm'], marker='o', s=100, label='rMax_MM')

    # Plot options
    if publication:
        pubPMF(pairInfo, label="GB**")
    else:
        plt.title(sysTitle + " GB**")
        plt.xlabel("Distance (nm)")
        plt.ylabel("PMF (kcal/mol)")
        plt.ylim(qYlimDic.get(qprod))
        plt.xlim(xLim)
        plt.legend()
    
def graphMMX(pairInfo, publication=False):
    sysTitle = pairInfo['A']['name'] + " -- " + pairInfo['B']['name']
    qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

    # Check that data is available
    corrData = pairInfo.get('mmx', {}).get('diffData')
    if corrData is None: print("No MM* data available for this pair.") ; return

    # Load base data
    baseData = pairInfo['mm']['pmfData']
    # Load transition radii
    dft_rMax = pairInfo.get('mmx', {}).get('dft_rMax')
    dft_rMax = (dft_rMax, getY(pairInfo['dft']['pmfData'], dft_rMax))
    mm_rMin = pairInfo.get('mmx', {}).get('mm_rMin')
    mm_rMin = (mm_rMin, getY(pairInfo['mm']['pmfData'], mm_rMin))
    mm_rMax = pairInfo.get('gbx', {}).get('mm_rMax')
    mm_rMax = (mm_rMax, getY(pairInfo['mm']['pmfData'], mm_rMax))
    # load reference data
    ref1Data = pairInfo['dft']['pmfData']
    coulombData = coulombZero(baseData[:,0], qprod, mm_rMax[0])
    # trim reference data to end at corresponding rMax
    ref1Data = ref1Data[ref1Data[:,0] <= dft_rMax[0]]
    # Align GB* correction with base data
    corrData = pairInfo['mmx']['diffData']
    corrNewR = interpR(corrData, baseData[:,0])
    basePlusCorr = baseData[:,1] + corrNewR[:,1]

    # Plot MM* correction
    plt.fill_between(corrNewR[:,0], baseData[:,1], 
                     basePlusCorr, color=colorDic['mmx'], 
                     alpha=fillOpacity, label='MM* Correction') 
    # Plot reference data
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='DFT PMF', color=colorDic['dft'])
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--')
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='MM PMF', color=colorDic['mm'])
    # Plot transition radii
    plt.scatter(dft_rMax[0], dft_rMax[1], color=colorDic['dft'], marker='o', s=100, label='rMax_DFT')
    plt.scatter(mm_rMin[0], mm_rMin[1], color=colorDic['mm'], marker='o', s=100, label='rMin_MM')

    # Plot options
    if publication:
        pubPMF(pairInfo, label="MM*")
    else:
        plt.title(sysTitle + " MM*")
        plt.xlabel("Distance (nm)")
        plt.ylabel("PMF (kcal/mol)")
        plt.ylim(qYlimDic.get(qprod))
        plt.xlim(xLim)
        plt.legend()
   
def graphCorrectionPanels(pairInfo, save=None):
    fig, axs = plt.subplots(
        1, 3,
        figsize=(7.2, 2.35),
        sharex=True,
        sharey=True,
    )

    plt.sca(axs[0])
    graphGBX(pairInfo, publication=True)
    pubPMF(pairInfo, label="GB*", show_ylabel=True)

    plt.sca(axs[1])
    graphGBXX(pairInfo, publication=True)
    pubPMF(pairInfo, label="GB**", show_ylabel=False)

    plt.sca(axs[2])
    graphMMX(pairInfo, publication=True)
    pubPMF(pairInfo, label="MM*", show_ylabel=False)

    fig.tight_layout(w_pad=0.5)

    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=600)

    return fig, axs
    return fig, axs