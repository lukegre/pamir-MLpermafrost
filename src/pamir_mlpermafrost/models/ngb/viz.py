import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt


def get_feature_importance(model, names=None):
    import numpy as np
    
    if names is None:
        names = getattr(model, 'feature_names_in_')

    fi = np.array(model.feature_importances_, ndmin=2)
    nrows, ncols = fi.shape
    if nrows < ncols:
        fi = fi.T

    if min(nrows, ncols) == 1:
        columns = ['feature_importance']
    elif min(nrows, ncols) == 2:
        columns = ['feature_importance_loc', 'feature_importance_scale']
    fi = pd.DataFrame(fi, columns=columns, index=names)
    
    return fi


def plot_feature_importance(model, feature_names=None):

    feature_importances = get_feature_importance(model, feature_names)
    
    ncol = feature_importances.shape[1]
    fig_h = 4 
    fig_w = fig_h * 1.375 * ncol
    fig, axs = plt.subplots(1, ncol, figsize=[fig_w, fig_h], sharex=True, squeeze=False)
    axs = axs.flatten()

    for i, key in enumerate(feature_importances):
        ser = feature_importances[key].sort_values()
        ser.plot.barh(ax=axs[i], color=f'C{i}')
        axs[i].set_title(key, loc='left')
        axs[i].set_xlabel('Fractional importance')
        
    fig.tight_layout()
    
    p0 = axs[0].get_position()
    fig.suptitle('Feature importances', x=p0.x0, y=p0.y1 + 0.15, ha='left', weight='bold', size='xx-large')

    return fig, axs


    
def plot_train_test_split(data):
    props = dict(x='x', y='y', alpha=0.5, linewidth=0)
    ax = data.train.x.reset_index().plot.scatter(label='Train', c='b', **props)
    ax = data.test .x.reset_index().plot.scatter(label='Test',  c='r', ax=ax, **props)
    ax = data.valid.x.reset_index().plot.scatter(label='Valid', c='g', ax=ax, **props)
    ax.legend(loc='lower left', framealpha=1)
    
    return ax.figure


def plot_ground_temp_map(yhat: xr.DataArray, yhat_std: xr.DataArray, name=''):
    fig, axs = plt.subplots(1, 2, figsize=[11, 6], sharey=True, dpi=150)
    
    cbar_kwargs = dict(location='bottom', extend='both', pad=0.08)
    
    img = yhat.plot.imshow(vmin=-10, vmax=10, center=0, cmap='RdBu_r', ax=axs[0], cbar_kwargs=cbar_kwargs)
    img.axes.set_xlabel('')
    img.axes.set_ylabel('')
    img.axes.set_title(f'a) $\\mathbf{{\\mu}}$', loc='left', size='x-large')
    img.colorbar.set_label('Temperature [°C]')
    
    img = yhat_std.plot.imshow(robust=True, ax=axs[1], vmin=0, levels=np.arange(0, 4.6, 0.5), extend='max', cbar_kwargs=(cbar_kwargs|dict(extend='max')))
    img.axes.set_ylabel('')
    img.axes.set_xlabel('')
    img.axes.set_title(f'b) $\\mathbf{{\\sigma}}$', loc='left', size='x-large')
    img.colorbar.set_label('Temperature [°C]')
    
    fig.tight_layout()
    p0 = axs[0].get_position()
    fig.suptitle(f'Ground temperature estimates {name}', x=p0.x0, y=p0.y1 + 0.1, ha='left', weight='bold', size='xx-large')
    plt.show()

    return fig
    