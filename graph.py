import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import yaml

rcPath='/Users/yasmene/rc/'

################################### hard coded ###################################
colorDic = {
    'coulomb': '#9A9A9A',

    # base model blocks
    'dft':  "#9E3F3F",
    'mm':   '#CFAF00', #'#E0B72F',
    'gb':   '#2B6FA3',

    # correction arrows
    'mmx':  "#ff8400", #"#E9894E",
    'gbx':  '#2CA6A4',
    'gbxx': "#9B6BE8",
}

qYlimDic = {-2: (-4.5, 5.5),
            -1: (-4.5, 5.5),
             0: (-3, 7),
             1: (-3, 7),
             2: (-1, 9),
             4: (-1, 9),}

xLim = (0.1, 1.3)
fillOpacity = 0.5
lineThickness = 1.6

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


def correctedPMF(baseData, corrData):
    """Calculate the corrected PMF on the base PMF's distance grid."""
    correction = np.interp(
        baseData[:, 0],
        corrData[:, 0],
        corrData[:, 1],
        left=0.0,
        right=0.0,
    )
    return baseData[:, 0], baseData[:, 1] + correction

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

    # for line in ax.lines:
    #     line.set_linewidth(lineThickness)

    # Only resize scatter markers, NOT fill_between collections
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            coll.set_sizes([40])

    # remove legends in publication plots
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


################################### Graph Correction Construction ###################################


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

    # Plot Coulomb reference
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--', linewidth=lineThickness)
    # Plot GB* correction
    plt.fill_between(corrNewR[:,0], baseData[:,1], 
                     basePlusCorr, color=colorDic['gbx'], 
                     alpha=fillOpacity, label='GB* Correction') 
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='GB PMF', color=colorDic['gb'], linewidth=lineThickness)
    # Plot reference data
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='MM PMF', color=colorDic['mm'], linewidth=lineThickness)
     # Plot transition radii
    plt.scatter(mm_rMax[0], mm_rMax[1], color=colorDic['mm'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMax_MM')
    
    # Plot options
    if publication:
        pubPMF(pairInfo)
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
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='MM PMF', color=colorDic['mm'], linewidth=lineThickness)
    plt.plot(ref2Data[:,0], ref2Data[:,1], label='DFT PMF', color=colorDic['dft'], linewidth=lineThickness)
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--', linewidth=lineThickness)
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='GB PMF', color=colorDic['gb'], linewidth=lineThickness)
    # Plot transition radii
    plt.scatter(dft_rMax[0], dft_rMax[1], color=colorDic['dft'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMax_DFT')
    plt.scatter(mm_rMin[0], mm_rMin[1], color=colorDic['mm'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMin_MM')
    plt.scatter(mm_rMax[0], mm_rMax[1], color=colorDic['mm'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMax_MM')

    # Plot options
    if publication:
        pubPMF(pairInfo)
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
    plt.plot(ref1Data[:,0], ref1Data[:,1], label='DFT PMF', color=colorDic['dft'], linewidth=lineThickness)
    plt.plot(coulombData[:,0], coulombData[:,1], label='Coulomb', color=colorDic['coulomb'], linestyle='--', linewidth=lineThickness)
    # Plot base data
    plt.plot(baseData[:,0], baseData[:,1], label='MM PMF', color=colorDic['mm'], linewidth=lineThickness)
    # Plot transition radii
    plt.scatter(dft_rMax[0], dft_rMax[1], color=colorDic['dft'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMax_DFT')
    plt.scatter(mm_rMin[0], mm_rMin[1], color=colorDic['mm'], marker='o', s=100, edgecolors='white', zorder = 10, label='rMin_MM')

    # Plot options
    if publication:
        pubPMF(pairInfo)
    else:
        plt.title(sysTitle + " MM*")
        plt.xlabel("Distance (nm)")
        plt.ylabel("PMF (kcal/mol)")
        plt.ylim(qYlimDic.get(qprod))
        plt.xlim(xLim)
        plt.legend()
   
def graphCorrectionPanels(pairInfo, save=None):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig, axs = plt.subplots(
        1, 3,
        figsize=(8, 2.7),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )

    plt.sca(axs[0])
    graphGBX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=True)

    plt.sca(axs[1])
    graphGBXX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=False)

    plt.sca(axs[2])
    graphMMX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=False)

    fig.tight_layout(w_pad=0.5)

    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")

    return fig, axs


################################### Graph Correction Construction ###################################


def graphGBX(pairInfo, publication=False):
    sysTitle = pairInfo['A']['name'] + " -- " + pairInfo['B']['name']
    qprod = pairInfo['A']['charge'] * pairInfo['B']['charge']

    corrData = pairInfo.get('gbx', {}).get('diffData')
    if corrData is None:
        print("No GB* data available for this pair.")
        return

    baseData = pairInfo['gb']['pmfData']
    mmData = pairInfo['mm']['pmfData']
    mm_rMax = pairInfo['gbx']['mm_rMax']
    mm_rMaxPoint = (mm_rMax, getY(mmData, mm_rMax))

    coulombData = coulombZero(baseData[:, 0], qprod, mm_rMax)
    mmReference = mmData[mmData[:, 0] <= mm_rMax]

    corrNewR = interpR(corrData, baseData[:, 0])
    basePlusCorr = baseData[:, 1] + corrNewR[:, 1]

    # Shade only through r_MM,max.
    shadeMask = corrNewR[:, 0] <= mm_rMax

    plt.plot(
        coulombData[:, 0], coulombData[:, 1],
        label='Coulomb',
        color=colorDic['coulomb'],
        linestyle='--',
        linewidth=lineThickness,
    )
    plt.fill_between(
        corrNewR[shadeMask, 0],
        baseData[shadeMask, 1],
        basePlusCorr[shadeMask],
        color=colorDic['gbx'],
        alpha=fillOpacity,
        label='GB* Correction',
    )
    plt.plot(
        baseData[:, 0], baseData[:, 1],
        label='GB PMF',
        color=colorDic['gb'],
        linewidth=lineThickness,
    )
    plt.plot(
        mmReference[:, 0], mmReference[:, 1],
        label='MM PMF',
        color=colorDic['mm'],
        linewidth=lineThickness,
    )
    plt.scatter(
        mm_rMaxPoint[0], mm_rMaxPoint[1],
        color=colorDic['mm'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMax_MM',
    )

    if publication:
        pubPMF(pairInfo)
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

    corrData = pairInfo.get('gbxx', {}).get('diffData')
    if corrData is None:
        print("No GB** data available for this pair.")
        return

    baseData = pairInfo['gb']['pmfData']
    mmData = pairInfo['mm']['pmfData']
    dftData = pairInfo['dft']['pmfData']

    dft_rMax = pairInfo['gbxx']['dft_rMax']
    mm_rMin = pairInfo['gbxx']['mm_rMin']
    mm_rMax = pairInfo['gbxx']['mm_rMax']

    dft_rMaxPoint = (dft_rMax, getY(dftData, dft_rMax))
    mm_rMinPoint = (mm_rMin, getY(mmData, mm_rMin))
    mm_rMaxPoint = (mm_rMax, getY(mmData, mm_rMax))

    mmReference = mmData[mmData[:, 0] <= mm_rMax]
    dftReference = dftData[dftData[:, 0] <= dft_rMax]
    coulombData = coulombZero(baseData[:, 0], qprod, mm_rMax)

    corrNewR = interpR(corrData, baseData[:, 0])
    basePlusCorr = baseData[:, 1] + corrNewR[:, 1]

    # Shade only through r_MM,max.
    shadeMask = corrNewR[:, 0] <= mm_rMax

    plt.fill_between(
        corrNewR[shadeMask, 0],
        baseData[shadeMask, 1],
        basePlusCorr[shadeMask],
        color=colorDic['gbxx'],
        alpha=fillOpacity,
        label='GB** Correction',
    )
    plt.plot(
        mmReference[:, 0], mmReference[:, 1],
        label='MM PMF',
        color=colorDic['mm'],
        linewidth=lineThickness,
    )
    plt.plot(
        dftReference[:, 0], dftReference[:, 1],
        label='DFT PMF',
        color=colorDic['dft'],
        linewidth=lineThickness,
    )
    plt.plot(
        coulombData[:, 0], coulombData[:, 1],
        label='Coulomb',
        color=colorDic['coulomb'],
        linestyle='--',
        linewidth=lineThickness,
    )
    plt.plot(
        baseData[:, 0], baseData[:, 1],
        label='GB PMF',
        color=colorDic['gb'],
        linewidth=lineThickness,
    )
    plt.scatter(
        dft_rMaxPoint[0], dft_rMaxPoint[1],
        color=colorDic['dft'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMax_DFT',
    )
    plt.scatter(
        mm_rMinPoint[0], mm_rMinPoint[1],
        color=colorDic['mm'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMin_MM',
    )
    plt.scatter(
        mm_rMaxPoint[0], mm_rMaxPoint[1],
        color=colorDic['mm'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMax_MM',
    )

    if publication:
        pubPMF(pairInfo)
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

    corrData = pairInfo.get('mmx', {}).get('diffData')
    if corrData is None:
        print("No MM* data available for this pair.")
        return

    baseData = pairInfo['mm']['pmfData']
    dftData = pairInfo['dft']['pmfData']

    dft_rMax = pairInfo['mmx']['dft_rMax']
    mm_rMin = pairInfo['mmx']['mm_rMin']
    mm_rMax = pairInfo['gbx']['mm_rMax']

    dft_rMaxPoint = (dft_rMax, getY(dftData, dft_rMax))
    mm_rMinPoint = (mm_rMin, getY(baseData, mm_rMin))

    dftReference = dftData[dftData[:, 0] <= dft_rMax]
    coulombData = coulombZero(baseData[:, 0], qprod, mm_rMax)

    corrNewR = interpR(corrData, baseData[:, 0])
    basePlusCorr = baseData[:, 1] + corrNewR[:, 1]

    # Shade only through r_MM,max.
    shadeMask = corrNewR[:, 0] <= mm_rMax

    plt.fill_between(
        corrNewR[shadeMask, 0],
        baseData[shadeMask, 1],
        basePlusCorr[shadeMask],
        color=colorDic['mmx'],
        alpha=fillOpacity,
        label='MM* Correction',
    )
    plt.plot(
        dftReference[:, 0], dftReference[:, 1],
        label='DFT PMF',
        color=colorDic['dft'],
        linewidth=lineThickness,
    )
    plt.plot(
        coulombData[:, 0], coulombData[:, 1],
        label='Coulomb',
        color=colorDic['coulomb'],
        linestyle='--',
        linewidth=lineThickness,
    )
    plt.plot(
        baseData[:, 0], baseData[:, 1],
        label='MM PMF',
        color=colorDic['mm'],
        linewidth=lineThickness,
    )
    plt.scatter(
        dft_rMaxPoint[0], dft_rMaxPoint[1],
        color=colorDic['dft'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMax_DFT',
    )
    plt.scatter(
        mm_rMinPoint[0], mm_rMinPoint[1],
        color=colorDic['mm'],
        marker='o', s=100, edgecolors='white',
        zorder=10, label='rMin_MM',
    )

    if publication:
        pubPMF(pairInfo)
    else:
        plt.title(sysTitle + " MM*")
        plt.xlabel("Distance (nm)")
        plt.ylabel("PMF (kcal/mol)")
        plt.ylim(qYlimDic.get(qprod))
        plt.xlim(xLim)
        plt.legend()


def graphCorrectionPanels(pairInfo, save=None):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig, axs = plt.subplots(
        1, 3,
        figsize=(8, 2.7),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0, 'hspace': 0},
    )

    plt.sca(axs[0])
    graphGBX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=True)

    plt.sca(axs[1])
    graphGBXX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=False)

    plt.sca(axs[2])
    graphMMX(pairInfo, publication=True)
    pubPMF(pairInfo, show_ylabel=False)

    fig.tight_layout(w_pad=0.5)

    if save is not None:
        fig.savefig(save, format='pdf', bbox_inches='tight')

    return fig, axs



################################### Full pair figure ###################################

from matplotlib.lines import Line2D


def _formatRadius(value):
    return "—" if value is None else f"{value:.3f} nm"


def _correctionRange(pairInfo, key):
    data = pairInfo.get(key, {}).get('diffData')
    if data is None:
        return "—"
    return f"{np.min(data[:, 0]):.3f}–{np.max(data[:, 0]):.3f} nm"


def _chargedName(species):
    """Format a species name with its charge in parentheses."""
    name = species['name']
    charge = species['charge']

    if charge == 0:
        return name

    sign = '+' if charge > 0 else '−'
    magnitude = abs(charge)
    chargeText = sign if magnitude == 1 else f"{magnitude}{sign}"

    return f"{name} ({chargeText})"


def _columnStatus(pairInfo, corrKey):
    """Return None when a column can be drawn, otherwise its message."""
    if corrKey in ('gbxx', 'mmx'):
        if pairInfo.get('dft', {}).get('pmfData') is None:
            return "No DFT data"

    if pairInfo.get(corrKey, {}).get('diffData') is None:
        return "No correction data"

    if corrKey in ('gbx', 'gbxx'):
        if pairInfo.get('gb', {}).get('pmfData') is None:
            return "No GB data"
        if pairInfo.get('mm', {}).get('pmfData') is None:
            return "No MM data"

    if corrKey == 'mmx':
        if pairInfo.get('mm', {}).get('pmfData') is None:
            return "No MM data"

    return None


def _blankPanel(ax, message):
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=10, color='0.45',
    )


def _drawResultPMF(ax, pairInfo, baseKey, corrKey,
                   dftPlotExtension=0.05):
    """Draw one resulting PMF panel."""
    base = pairInfo[baseKey]['pmfData']
    corr = pairInfo[corrKey]['diffData']
    r, corrected = correctedPMF(base, corr)

    # Corrected PMF: wide, transparent, and behind other curves
    ax.plot(
        r, corrected,
        color=colorDic[corrKey],
        linewidth=3.0,
        alpha=0.75,
        zorder=1,
    )

    # Base PMF: solid
    ax.plot(
        base[:, 0], base[:, 1],
        color=colorDic[baseKey],
        linewidth=lineThickness,
        zorder=2,
    )

    # MM reference: dotted; retain its low-r data
    if corrKey in ('gbx', 'gbxx'):
        mm = pairInfo['mm']['pmfData']
        mm_rMax = pairInfo[corrKey]['mm_rMax']
        mm = mm[mm[:, 0] <= mm_rMax]

        ax.plot(
            mm[:, 0], mm[:, 1],
            color=colorDic['mm'],
            linewidth=lineThickness,
            linestyle=':',
            dash_capstyle='round',
            zorder=4,
        )

    # DFT reference: dotted; extend beyond construction radius
    if corrKey in ('gbxx', 'mmx'):
        dft = pairInfo['dft']['pmfData']
        dft_rMax = pairInfo[corrKey]['dft_rMax']
        dft = dft[dft[:, 0] <= dft_rMax + dftPlotExtension]

        ax.plot(
            dft[:, 0], dft[:, 1],
            color=colorDic['dft'],
            linewidth=lineThickness,
            linestyle=':',
            dash_capstyle='round',
            zorder=5,
        )


def graphFullPairPanel(pairInfo, save=None, structureImage=None,
                       dftPlotExtension=0.05):
    """
    Pair title and legend, structure placeholder, correction
    construction, resulting PMFs, and matching radii.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(9.2, 7.7))

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[5.65, 0.85],
        hspace=0.145,
        left=0.13, right=0.97, top=0.93, bottom=0.06,
    )
    main = outer[0].subgridspec(
        2, 1,
        height_ratios=[1.20, 4.45],
        hspace=0.14,
    )
    header = main[0].subgridspec(
        2, 2,
        height_ratios=[0.40, 0.60],
        width_ratios=[3.0, 0.85],
        hspace=0.03,
        wspace=0.16,
    )
    plotGrid = main[1].subgridspec(
        2, 3, wspace=0, hspace=0
    )
    infoGrid = outer[1].subgridspec(
        1, 3, wspace=0.10
    )

    # Title
    axTitle = fig.add_subplot(header[0, 0])
    axTitle.axis('off')

    pairName = (
        f"{_chargedName(pairInfo['A'])}"
        " – "
        f"{_chargedName(pairInfo['B'])}"
    )
    axTitle.text(
        0, 0.65, pairName,
        transform=axTitle.transAxes,
        fontsize=15, fontweight='bold',
        ha='left', va='center',
    )

    # Legend
    axLegend = fig.add_subplot(header[1, 0])
    axLegend.axis('off')

    # Input order gives GB, MM, DFT, Coulomb across the first row
    # and GB*, GB**, MM* across the second.
    legendItems = [
        Line2D([0], [0], color=colorDic['gb'],
               lw=lineThickness, label='GB'),
        Line2D([0], [0], color=colorDic['gbx'],
               lw=3.0, alpha=0.75, label='GB*'),
        Line2D([0], [0], color=colorDic['mm'],
               lw=lineThickness, label='MM'),
        Line2D([0], [0], color=colorDic['gbxx'],
               lw=3.0, alpha=0.75, label='GB**'),
        Line2D([0], [0], color=colorDic['dft'],
               lw=lineThickness, label='DFT'),
        Line2D([0], [0], color=colorDic['mmx'],
               lw=3.0, alpha=0.75, label='MM*'),
        Line2D([0], [0], color=colorDic['coulomb'],
               lw=lineThickness, linestyle=':', label='Coulomb'),
    ]

    axLegend.legend(
        handles=legendItems,
        loc='center left',
        bbox_to_anchor=(0.08, 0.5),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.25,
        labelspacing=0.35,
        borderaxespad=0,
    )

    # Structure placeholder in upper-right corner
    axStructure = fig.add_subplot(header[:, 1])
    axStructure.set_xticks([])
    axStructure.set_yticks([])

    for spine in axStructure.spines.values():
        spine.set_color('0.75')

    if structureImage is None:
        axStructure.text(
            0.5, 0.5, "Pair structure\nplaceholder",
            transform=axStructure.transAxes,
            ha='center', va='center',
            fontsize=9, color='0.45',
        )
    else:
        axStructure.imshow(plt.imread(structureImage))
        axStructure.set_aspect('equal')

    # Continuous 2 × 3 grid
    constructionAxes = [fig.add_subplot(plotGrid[0, 0])]
    constructionAxes += [
        fig.add_subplot(
            plotGrid[0, i],
            sharex=constructionAxes[0],
            sharey=constructionAxes[0],
        )
        for i in range(1, 3)
    ]
    resultAxes = [
        fig.add_subplot(
            plotGrid[1, i],
            sharex=constructionAxes[i],
            sharey=constructionAxes[0],
        )
        for i in range(3)
    ]

    columns = [
        ('GB*',  'gb', 'gbx',  graphGBX),
        ('GB**', 'gb', 'gbxx', graphGBXX),
        ('MM*',  'mm', 'mmx',  graphMMX),
    ]

    for i, (title, baseKey, corrKey, constructionFn) in enumerate(columns):
        status = _columnStatus(pairInfo, corrKey)

        constructionAx = constructionAxes[i]
        plt.sca(constructionAx)

        if status is None:
            constructionFn(pairInfo, publication=True)
            for line in constructionAx.lines:
                if line.get_label() == 'Coulomb':
                    line.set_linestyle(':')
        else:
            _blankPanel(constructionAx, status)

        pubPMF(pairInfo, show_ylabel=(i == 0))
        constructionAx.set_title(
            title, fontsize=13, fontweight='bold', pad=5
        )

        resultAx = resultAxes[i]
        plt.sca(resultAx)

        if status is None:
            _drawResultPMF(
                resultAx, pairInfo, baseKey, corrKey,
                dftPlotExtension=dftPlotExtension,
            )
        else:
            _blankPanel(resultAx, status)

        pubPMF(pairInfo, show_ylabel=(i == 0))

    # Show distance labels only on the bottom row
    for ax in constructionAxes:
        ax.set_xlabel("")
        ax.tick_params(axis='x', which='both', labelbottom=False)

    constructionAxes[0].set_ylabel(
        r"(kcal mol$^{-1}$)",
        fontsize=10, fontweight='normal', labelpad=8,
    )
    resultAxes[0].set_ylabel(
        r"(kcal mol$^{-1}$)",
        fontsize=10, fontweight='normal', labelpad=8,
    )

    constructionAxes[0].text(
        -0.28, 0.5,
        r"$\mathbf{\Delta U}$ construction",
        transform=constructionAxes[0].transAxes,
        rotation=90,
        ha='center', va='center',
        fontsize=10, fontweight='bold',
    )
    resultAxes[0].text(
        -0.28, 0.5,
        "Resulting PMFs",
        transform=resultAxes[0].transAxes,
        rotation=90,
        ha='center', va='center',
        fontsize=10, fontweight='bold',
    )

    radiusLabels = {
        'gbx': [
            r"$r_{\mathrm{MM,max}}$",
        ],
        'gbxx': [
            r"$r_{\mathrm{DFT,max}}$",
            r"$r_{\mathrm{MM,min}}$",
            r"$r_{\mathrm{MM,max}}$",
        ],
        'mmx': [
            r"$r_{\mathrm{DFT,max}}$",
            r"$r_{\mathrm{MM,min}}$",
        ],
    }
    radiusKeys = {
        'gbx':  ['mm_rMax'],
        'gbxx': ['dft_rMax', 'mm_rMin', 'mm_rMax'],
        'mmx':  ['dft_rMax', 'mm_rMin'],
    }

    for i, (_, _, corrKey, _) in enumerate(columns):
        axInfo = fig.add_subplot(infoGrid[i])
        axInfo.axis('off')

        status = _columnStatus(pairInfo, corrKey)
        if status is not None:
            entries = [status]
        else:
            entries = [
                f"{label} = "
                f"{_formatRadius(pairInfo[corrKey].get(key))}"
                for label, key in zip(
                    radiusLabels[corrKey],
                    radiusKeys[corrKey],
                )
            ]
            entries.append(
                f"Correction data: {_correctionRange(pairInfo, corrKey)}"
            )

        axInfo.text(
            0.02, 0.95, "\n".join(entries),
            transform=axInfo.transAxes,
            ha='left', va='top',
            fontsize=8.5, linespacing=1.45,
        )

    if save is not None:
        fig.savefig(save, format='pdf', bbox_inches='tight')

    return fig, constructionAxes, resultAxes



################################### Grouped ΔU construction panels ###################################


def _pairType(pairInfo):
    """Classify pairs using species abbreviations from about.yml."""
    abbreviations = {
        pairInfo['A']['abbreviation'].strip().lower(),
        pairInfo['B']['abbreviation'].strip().lower(),
    }

    # A DMA–acetate pair belongs to the DMA group.
    if 'dma' in abbreviations:
        return 'DMA–anything'
    if 'ace' in abbreviations:
        return 'acetate–anything'
    return 'ion–ion'


def _formatQprod(qprod):
    return f"{qprod:+g}" if qprod != 0 else "0"


def _drawStackedPairName(fig, pairInfo, yCenter):
    """Center name A, a vertical bar, and name B around a row."""
    figureHeight = fig.get_size_inches()[1]

    # Convert a fixed physical spacing to figure coordinates.
    offset = 0.19 / figureHeight

    fig.text(
        0.04, yCenter + offset,
        _chargedName(pairInfo['A']),
        ha='center', va='center',
        fontsize=9, fontweight='bold',
    )
    fig.text(
        0.04, yCenter,
        "│",
        ha='center', va='center',
        fontsize=9,
    )
    fig.text(
        0.04, yCenter - offset,
        _chargedName(pairInfo['B']),
        ha='center', va='center',
        fontsize=9, fontweight='bold',
    )


def _constructionFigure(groupPairs, heading):
    """Make one compact construction figure for a list of pairs."""
    nRows = len(groupPairs)

    figureHeight = 2.45 if nRows == 1 else 1.65 * nRows + 0.6
    fig = plt.figure(figsize=(9.2, figureHeight))

    # Extra heading space for a one-row figure
    plotTop = 0.76 if nRows == 1 else 0.88

    grid = fig.add_gridspec(
        nRows, 3,
        left=0.21, right=0.97,
        top=plotTop, bottom=0.12,
        wspace=0, hspace=0,
    )

    allAxes = []
    sharedX = None

    columns = [
        ('GB*',  'gbx',  graphGBX),
        ('GB**', 'gbxx', graphGBXX),
        ('MM*',  'mmx',  graphMMX),
    ]

    for row, pairInfo in enumerate(groupPairs):
        firstAx = fig.add_subplot(
            grid[row, 0],
            sharex=sharedX,
        )
        if sharedX is None:
            sharedX = firstAx

        rowAxes = [firstAx]
        rowAxes += [
            fig.add_subplot(
                grid[row, col],
                sharex=sharedX,
                sharey=firstAx,
            )
            for col in (1, 2)
        ]
        allAxes.append(rowAxes)

        for col, (title, corrKey, graphFn) in enumerate(columns):
            ax = rowAxes[col]
            plt.sca(ax)

            status = _columnStatus(pairInfo, corrKey)

            if status is None:
                graphFn(pairInfo, publication=True)

                for line in ax.lines:
                    if line.get_label() == 'Coulomb':
                        line.set_linestyle(':')
            else:
                _blankPanel(ax, status)

            pubPMF(pairInfo, show_ylabel=(col == 0))

            if row == 0:
                ax.set_title(
                    title, fontsize=12,
                    fontweight='bold', pad=5,
                )
            else:
                ax.set_title("")

        firstAx.set_ylabel(
            r"Energy (kcal mol$^{-1}$)",
            fontsize=8,
            labelpad=6,
        )

        box = firstAx.get_position()
        yCenter = (box.y0 + box.y1) / 2
        _drawStackedPairName(fig, pairInfo, yCenter)

    # Only the bottom row displays the shared distance axis.
    for rowAxes in allAxes[:-1]:
        for ax in rowAxes:
            ax.set_xlabel("")
            ax.tick_params(
                axis='x', which='both',
                labelbottom=False,
            )

    for ax in allAxes[-1]:
        ax.set_xlabel(r"$r$ (nm)", fontsize=9)
        ax.tick_params(
            axis='x', which='both',
            labelbottom=True,
        )

    fig.suptitle(
        heading,
        fontsize=13, fontweight='bold',
        y=0.98,
    )

    return fig, allAxes


def iterGroupedConstructions(pairInfos):
    """
    Yield one figure at a time:
      1. DMA–anything, acetate–anything, ion–ion.
      2. Each charge product.

    Yields (grouping, groupName, fig, axes).
    """
    byType = {}
    byQprod = {}

    for pairInfo in pairInfos:
        pairType = _pairType(pairInfo)
        qprod = (
            pairInfo['A']['charge'] *
            pairInfo['B']['charge']
        )

        byType.setdefault(pairType, []).append(pairInfo)
        byQprod.setdefault(qprod, []).append(pairInfo)

    for pairType in (
        'DMA–anything',
        'acetate–anything',
        'ion–ion',
    ):
        if pairType in byType:
            fig, axes = _constructionFigure(
                byType[pairType],
                heading=f"ΔU construction | {pairType}",
            )
            yield 'type', pairType, fig, axes

    for qprod in sorted(byQprod):
        fig, axes = _constructionFigure(
            byQprod[qprod],
            heading=(
                "ΔU construction | "
                f"qA qB = {_formatQprod(qprod)}"
            ),
        )
        yield 'qprod', qprod, fig, axes



#################################### Group by Model ###################################


# def graphCorrectionGrid(pairInfos, layout, model='gbx', rowYlims=None):
#     """
#     Plot correction panels in the supplied layout.

#     model: 'gbx', 'gbxx', or 'mmx'
#     rowYlims: optional list of (minimum, maximum), one per row
#     """
#     model = model.lower()
#     if model not in {'gbx', 'gbxx', 'mmx'}:
#         raise ValueError("model must be 'gbx', 'gbxx', or 'mmx'")

#     if not layout or not layout[0]:
#         raise ValueError("layout must contain at least one pair")

#     nRows = len(layout)
#     nCols = len(layout[0])
#     if any(len(row) != nCols for row in layout):
#         raise ValueError("Every layout row must have the same number of pairs")
#     if rowYlims is not None and len(rowYlims) != nRows:
#         raise ValueError("rowYlims must contain one range per row")

#     def pairKey(names):
#         return tuple(sorted(name.strip().lower() for name in names))

#     byPair = {}
#     for pair in pairInfos:
#         key = pairKey((
#             pair['A']['abbreviation'],
#             pair['B']['abbreviation'],
#         ))
#         if key in byPair:
#             raise ValueError(f"Duplicate pair: {key}")
#         byPair[key] = pair

#     missing = [
#         names for row in layout for names in row
#         if pairKey(names) not in byPair
#     ]
#     if missing:
#         raise ValueError(f"Pairs absent from pairInfos: {missing}")

#     graphFns = {
#         'gbx': graphGBX,
#         'gbxx': graphGBXX,
#         'mmx': graphMMX,
#     }

#     fig, axes = plt.subplots(
#         nRows, nCols,
#         figsize=(3.5 * nCols, 2.4 * nRows),
#         sharex=True,
#         sharey='row',
#         squeeze=False,
#         gridspec_kw={'wspace': 0, 'hspace': 0},
#     )
#     fig.subplots_adjust(
#         left=0.10, right=0.98, bottom=0.07, top=0.95,
#     )

#     for rowIndex, row in enumerate(layout):
#         for colIndex, names in enumerate(row):
#             pair = byPair[pairKey(names)]
#             ax = axes[rowIndex, colIndex]
#             plt.sca(ax)

#             # A YAML entry can exist without loaded correction data.
#             correction = pair.get(model) or {}
#             hasCorrection = correction.get('diffData') is not None

#             if model == 'gbxx' and not hasCorrection:
#                 # Show GB* where a DFT-based GB** correction is unavailable.
#                 graphGBX(pair, publication=True)
#             elif model == 'mmx' and not hasCorrection:
#                 # Leave the panel empty except for its explanatory message.
#                 ax.set_xlim(*xLim)
#             else:
#                 graphFns[model](pair, publication=True)

#             if not (model == 'mmx' and not hasCorrection):
#                 for line in ax.lines:
#                     if line.get_label() == 'Coulomb':
#                         line.set_linestyle(':')

#                 pubPMF(pair, show_ylabel=(colIndex == 0))

#             # Apply shared row limits after plotting.
#             if rowYlims is not None:
#                 ax.set_ylim(*rowYlims[rowIndex])

#             if model == 'mmx' and not hasCorrection:
#                 ax.text(
#                     0.5, 0.5, 'No DFT data',
#                     transform=ax.transAxes,
#                     ha='center', va='center',
#                     fontsize=10, color='0.4',
#                     zorder=20,
#                 )

#             displayNames = [
#                 'DMA' if name.lower() == 'dma' else name.capitalize()
#                 for name in names
#             ]
#             ax.text(
#                 0.97, 0.96,
#                 '–'.join(displayNames),
#                 transform=ax.transAxes,
#                 ha='right', va='top',
#                 fontsize=10, fontweight='bold',
#                 bbox=dict(
#                     facecolor='white',
#                     edgecolor='none',
#                     alpha=0.75,
#                     pad=1.5,
#                 ),
#                 zorder=21,
#             )

#             if rowIndex == nRows - 1:
#                 ax.set_xlabel(r'$r$ (nm)', fontsize=9)
#             else:
#                 ax.set_xlabel('')
#                 ax.tick_params(
#                     axis='x', which='both',
#                     labelbottom=False,
#                 )

#             if colIndex != 0:
#                 ax.set_ylabel('')

#     fig.suptitle(
#         {
#             'gbx': 'GB*',
#             'gbxx': 'GB**',
#             'mmx': 'MM*',
#         }[model] + ' corrections',
#         fontsize=14, fontweight='bold', y=0.995,
#     )
#     return fig, axes



def graphCorrectionGrid(pairInfos, layout, model='gbx', rowYlims=None):
    """
    Plot correction panels in the supplied layout.

    model: 'gbx', 'gbxx', or 'mmx'
    rowYlims: optional list of (minimum, maximum), one per row
    """
    model = model.lower()
    if model not in {'gbx', 'gbxx', 'mmx'}:
        raise ValueError("model must be 'gbx', 'gbxx', or 'mmx'")

    if not layout or not layout[0]:
        raise ValueError("layout must contain at least one pair")

    nRows = len(layout)
    nCols = len(layout[0])
    if any(len(row) != nCols for row in layout):
        raise ValueError("Every layout row must have the same number of pairs")
    if rowYlims is not None and len(rowYlims) != nRows:
        raise ValueError("rowYlims must contain one range per row")

    def pairKey(names):
        return tuple(sorted(name.strip().lower() for name in names))

    byPair = {}
    for pair in pairInfos:
        key = pairKey((
            pair['A']['abbreviation'],
            pair['B']['abbreviation'],
        ))
        if key in byPair:
            raise ValueError(f"Duplicate pair: {key}")
        byPair[key] = pair

    missing = [
        names for row in layout for names in row
        if pairKey(names) not in byPair
    ]
    if missing:
        raise ValueError(f"Pairs absent from pairInfos: {missing}")

    graphFns = {
        'gbx': graphGBX,
        'gbxx': graphGBXX,
        'mmx': graphMMX,
    }

    fig, axes = plt.subplots(
        nRows, nCols,
        figsize=(3.5 * nCols, 2.4 * nRows),
        sharex=True,
        sharey='row',
        squeeze=False,
        gridspec_kw={'wspace': 0, 'hspace': 0},
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.07, top=0.95,
    )

    for rowIndex, row in enumerate(layout):
        for colIndex, names in enumerate(row):
            pair = byPair[pairKey(names)]
            ax = axes[rowIndex, colIndex]
            plt.sca(ax)

            # Check whether correction data actually loaded.
            correction = pair.get(model) or {}
            hasCorrection = correction.get('diffData') is not None

            if model == 'gbxx' and not hasCorrection:
                # Display GB* when GB** has no DFT-based correction.
                graphGBX(pair, publication=True)
            elif model == 'mmx' and not hasCorrection:
                # Keep an empty panel with axes and an explanation.
                ax.set_xlim(*xLim)
            else:
                graphFns[model](pair, publication=True)

            if not (model == 'mmx' and not hasCorrection):
                for line in ax.lines:
                    if line.get_label() == 'Coulomb':
                        line.set_linestyle(':')

                pubPMF(pair, show_ylabel=(colIndex == 0))

            # pubPMF is skipped for an empty MM* panel, so add its
            # y-axis label explicitly when it is first in a row.
            if model == 'mmx' and not hasCorrection and colIndex == 0:
                ax.set_ylabel(
                    r'PMF (kcal mol$^{-1}$)',
                    fontsize=8,
                    labelpad=6,
                )

            if rowYlims is not None:
                ax.set_ylim(*rowYlims[rowIndex])

            if model == 'mmx' and not hasCorrection:
                ax.text(
                    0.5, 0.5, 'No DFT data',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=10, color='0.4',
                    zorder=20,
                )

            # Put the pair name inside the upper-right corner.
            displayNames = [
                'DMA' if name.lower() == 'dma' else name.capitalize()
                for name in names
            ]
            ax.text(
                0.97, 0.96,
                '–'.join(displayNames),
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.75,
                    pad=1.5,
                ),
                zorder=21,
            )

            if rowIndex == nRows - 1:
                ax.set_xlabel(r'$r$ (nm)', fontsize=9)
            else:
                ax.set_xlabel('')
                ax.tick_params(
                    axis='x', which='both',
                    labelbottom=False,
                )

            if colIndex != 0:
                ax.set_ylabel('')

    fig.suptitle(
        {
            'gbx': 'GB*',
            'gbxx': 'GB**',
            'mmx': 'MM*',
        }[model] + ' corrections',
        fontsize=14, fontweight='bold', y=0.995,
    )
    return fig, axes