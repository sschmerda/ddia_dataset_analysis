from .standard_import import *

### pandas options ###
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

### seaborn options ###
sns.set(rc = {'figure.figsize':(15,8)})
#sns.set_context('notebook', rc={'font.size':20,'axes.titlesize':30,'axes.labelsize':20, 'xtick.labelsize':16, 'ytick.labelsize':16, 'legend.fontsize':20, 'legend.title_fontsize':20})   
sns.set_style('darkgrid')
marker_config = {'marker':'o',
                 'markerfacecolor':'white', 
                 'markeredgecolor':'black',
                 'markersize':'10'}
marker_config_eval_metric_mean = {'marker':'o',
                                  'markerfacecolor':'red', 
                                  'markeredgecolor':'black',
                                  'markersize':'6'}
### numpy options ###
# turn off an unecessary numpy warning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
