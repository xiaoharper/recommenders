import matplotlib.pyplot as plt
import numpy as np


def bar_chart(
    values,
    labels,
    errors=None,
    x_name=None,
    y_name=None,
    bar_width=0.35,
):
    if errors:
        assert len(values) == len(errors)

    if isinstance(labels, str):
        values = [values]
        labels = [labels]
    else:
        assert len(values) == len(labels)
        
    ind = np.arange(len(values[0]))  # the x locations for the bar groups
    pos = (1-len(values[0])) * (bar_width/2)  # x location starting position
    
    fig, ax = plt.subplots()
    
    for i, v in enumerate(values):
        ax.bar(ind + pos, v, bar_width, yerr=errors[i] if errors else None,
               label=labels[i])
        pos += bar_width

    if x_name: ax.set_xlabel(x_name)
    if y_name: ax.set_ylabel(y_name)
    
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
