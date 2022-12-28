from IPython.display import display, Markdown, Latex
from lmfit.models import GaussianModel, ExponentialModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def printMD(message:str):
    display(Markdown(message))

def fit_my_data(w_markers_cropped, 
                g1_center:dict={"value":25, "min":20, "max":30}, 
                g1_sigma:dict={"value":3, "min":2}):
    # Data
    _axis = 0 # 0 - keeps x; 1 - keeps y
    y_markers_markers_cropped = range(w_markers_cropped.shape[0])   # Rows      (this, _)
    x_markers_markers_cropped = range(w_markers_cropped.shape[1])   # Columns   (_, this)
    w_markers_summed = w_markers_cropped.sum(axis=_axis)
    x_markers_summed = np.arange(len(w_markers_summed))

    # Model 
    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(w_markers_summed, x=x_markers_summed)

    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    pars['g1_center'].set(value=g1_center['value'], min=g1_center['min'], max=g1_center['max'])
    pars['g1_sigma'].set(value=g1_center['value'], min=g1_center['min'])

    mod = gauss1 + exp_mod
    init = mod.eval(pars, x=x_markers_summed)
    out = mod.fit(w_markers_summed, pars, x=x_markers_summed)

    # Printing section
    printMD("**Results report**")
    fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1]})
    # Colormesh
    ax[0].pcolormesh(x_markers_markers_cropped, y_markers_markers_cropped, w_markers_cropped, cmap="RdYlBu")
    plt.xticks(x_markers_markers_cropped, rotation=90)

    # Fitted graph
    ## Uncertainty
    dely = out.eval_uncertainty(sigma=3)
    ax[1].fill_between(x_markers_summed, 
                    out.best_fit-dely, 
                    out.best_fit+dely, 
                    color="#ABABAB",
                    label='3-$\sigma$ uncertainty band',
                    alpha=0.2)
    ## Graph itself
    ax[1].plot(x_markers_summed, out.data, 'o', label='Actual data')
    ax[1].scatter(x_markers_summed, out.best_fit, label='Fitted data', c='r', alpha=0.3, linewidths=0.1)
    ax[1].plot(x_markers_summed, out.best_fit, '--', label='Best fit')
    ax[1].set_yticks(np.linspace(out.data.min()-out.data.min()*0.01, 
                                 out.data.max()+out.data.max()*0.01, 
                                 7))
    ax[1].grid()
    ax[1].set_title("Fit (Explonential + Gaussian)")
    ax[1].legend()

    # Residuals
    out.plot_residuals(ax=ax[2], title="Residuals plot")
    ax[2].grid()
    ax[2].legend(['Fit line', 'epsilon'])

    plt.show()

    return out.params