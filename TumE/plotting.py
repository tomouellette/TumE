import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
from scipy import stats
import matplotlib.gridspec as gridspec
import string
import math
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
COLOR = 'black'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

"""
Plotting TumE predictions
"""

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, fontsize = 8, frameon=False, **kws)

def plot(tume_estimate, title = 'Sample', width = 10, height = 5, save_plot = False, save_dir = None, pdf = False, smoothing = 2):
    sns.set_style("white")
    m, ns, f, t, f1, f2, t1, t2 = tume_estimate['all_estimates']
    vaf, annotate = np.array(list(tume_estimate['annotated']['VAF'])), np.array(list(tume_estimate['annotated']['annotation']))
    # Parsimony check for 1 and 2 subclones (overlapping intervals)
    if (np.round(np.quantile(ns, q = 0.055, axis=0)[2] - np.quantile(ns, q = 0.945, axis=0)[1],2) <= 0) & (np.quantile(m.flatten(), q=0.055) > 0.5):
        explain_with_parsimony = True
        print(True)
    else:
        explain_with_parsimony = False
    if np.quantile(m.flatten(), q=0.055) <= 0.5:
        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        # Generate VAF distribution with clusters
        if (np.sum(annotate == 'Neutral tail') > 0) & (np.sum(annotate == 'Clonal') > 0):
            pal = ['#008EA0', '#CCCCCC']
        elif (np.sum(annotate == 'Neutral tail') == 0) & (np.sum(annotate == 'Clonal') > 0):
            pal = ['#CCCCCC']
        else:
            pal = ['#008EA0']

        sns.histplot(x = vaf, hue = annotate, palette = pal, kde = True, kde_kws={'bw_adjust': smoothing}, ax=ax, bins = 100)
        move_legend(ax, "upper right")
        ax.set_title(title, fontdict = {'fontsize':10}, loc='left')

        # Annotate P(Selection)
        ax.annotate(f'P(selection) {"{:.2f}".format(np.mean(m))} [{round(np.quantile(m, q=[0.055])[0],2)}, {round(np.quantile(m, q=[0.945])[0],2)}]', xy=(0.98,0.36), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        ax.annotate(f'P(0 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[0],2))} [{round(np.quantile(ns, q=0.055, axis=0)[0],2)}, {round(np.quantile(ns, q=0.945, axis=0)[0],2)}]', xy=(0.98,0.27), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        ax.annotate(f'P(1 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[1],2))} [{round(np.quantile(ns, q=0.055, axis=0)[1],2)}, {round(np.quantile(ns, q=0.945, axis=0)[1],2)}]', xy=(0.98,0.18), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        ax.annotate(f'P(2 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[2],2))} [{round(np.quantile(ns, q=0.055, axis=0)[2],2)}, {round(np.quantile(ns, q=0.945, axis=0)[2],2)}]', xy=(0.98,0.09), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.vlines(0.5, 0, 1, linestyle = 'dashed', transform=ax.get_xaxis_transform(), color = 'black', alpha = 1)

        if save_plot == True:
            fig.savefig(save_dir, dpi=600)

        return ax

    elif (np.argmax(np.mean(ns, axis=0)) == 1) or (explain_with_parsimony == True):
        colors = np.array(['#008EA0', '#C71100', '#CCCCCC', '#FF6F00', '#8A4198', '#5A9599', '#84D7E1'])
        rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for h in colors]

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[3,1]})
        fig.set_size_inches(width, height)

        # Generate VAF distribution with clusters
        pal ={"Neutral tail":rgb_colors[0], "Subclone":rgb_colors[1], "Clonal": rgb_colors[2]}
        sns.histplot(x = vaf, hue = annotate, palette = pal, kde = True, ax=axs[0], kde_kws={'bw_adjust':smoothing}, bins = 100)
        move_legend(axs[0], "upper right")
        axs[0].vlines(np.mean(f), 0, 1, linestyle = 'dashed', transform=axs[0].get_xaxis_transform(), color = colors[1], alpha = 1)
        axs[0].set_title(title, fontdict = {'fontsize':10}, loc='left')

        # Annotate P(Selection)
        axs[0].annotate(f'P(selection) {"{:.2f}".format(np.mean(m))} [{round(np.quantile(m, q=[0.055])[0],2)}, {round(np.quantile(m, q=[0.945])[0],2)}]', xy=(0.98,0.36), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(0 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[0],2))} [{round(np.quantile(ns, q=0.055, axis=0)[0],2)}, {round(np.quantile(ns, q=0.945, axis=0)[0],2)}]', xy=(0.98,0.27), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(1 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[1],2))} [{round(np.quantile(ns, q=0.055, axis=0)[1],2)}, {round(np.quantile(ns, q=0.945, axis=0)[1],2)}]', xy=(0.98,0.18), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(2 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[2],2))} [{round(np.quantile(ns, q=0.055, axis=0)[2],2)}, {round(np.quantile(ns, q=0.945, axis=0)[2],2)}]', xy=(0.98,0.09), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)

        # Generate approximate posterior panel
        sns.set_style("white")
        sns.histplot(x = f, ax=axs[1], bins = 15, edgecolor = '#CCCCCC', facecolor = (rgb_colors[1][0], rgb_colors[1][1], rgb_colors[1][2], 0.25))
        sns.kdeplot(x = f, ax=axs[1], color = colors[1])

        axs[1].vlines(np.mean(f), 0, 1, linestyle = 'dashed', transform=axs[1].get_xaxis_transform(), color = colors[1], alpha = 1)
        axs[0].vlines(0.5, 0, 1, linestyle = 'dashed', transform=axs[0].get_xaxis_transform(), color = 'black', alpha = 1)

        axs[1].set(xlabel='Variant Allele Frequency (VAF)', ylabel='Density')
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["left"].set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].annotate("Subclone frequency\napproximate\nposterior", xy=(1,1), xycoords="axes fraction",xytext=(0,-10), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[1].set_xlim(0, 1)
        axs[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        if save_plot == True:
            fig.savefig(save_dir, dpi=600)

        return axs

    elif np.argmax(np.mean(ns, axis=0)) == 2:
        colors = np.array(['#008EA0', '#C71100', '#FF6F00', '#CCCCCC', '#8A4198', '#5A9599', '#84D7E1'])
        rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for h in colors]

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[3,1]})
        fig.set_size_inches(width, height)

        # Generate VAF distribution with clusters
        pal ={"Neutral tail":rgb_colors[0], "Subclone B":rgb_colors[1], "Subclone A":rgb_colors[2], "Clonal": rgb_colors[3]}
        sns.histplot(x = vaf, hue = annotate, palette = pal, kde = True, kde_kws={'bw_adjust':smoothing}, ax=axs[0], bins = 100)
        move_legend(axs[0], "upper right")
        axs[0].vlines(np.mean(f1), 0, 1, linestyle = 'dashed', transform=axs[0].get_xaxis_transform(), color = colors[1], alpha = 1)
        axs[0].vlines(np.mean(f2), 0, 1, linestyle = 'dashed', transform=axs[0].get_xaxis_transform(), color = colors[2], alpha = 1)
        axs[0].set_title(title, fontdict = {'fontsize':10}, loc='left')

        # Annotate P(Selection)
        axs[0].annotate(f'P(selection) {"{:.2f}".format(np.mean(m))} [{round(np.quantile(m, q=[0.055])[0],2)}, {round(np.quantile(m, q=[0.945])[0],2)}]', xy=(0.98,0.36), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(0 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[0],2))} [{round(np.quantile(ns, q=0.055, axis=0)[0],2)}, {round(np.quantile(ns, q=0.945, axis=0)[0],2)}]', xy=(0.98,0.27), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(1 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[1],2))} [{round(np.quantile(ns, q=0.055, axis=0)[1],2)}, {round(np.quantile(ns, q=0.945, axis=0)[1],2)}]', xy=(0.98,0.18), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[0].annotate(f'P(2 subclone) {"{:.2f}".format(round(np.mean(ns, axis=0)[2],2))} [{round(np.quantile(ns, q=0.055, axis=0)[2],2)}, {round(np.quantile(ns, q=0.945, axis=0)[2],2)}]', xy=(0.98,0.09), xycoords="axes fraction",xytext=(0,0), textcoords="offset points",ha="right", va="top", fontsize = 8)

        # Generate approximate posterior panel
        sns.set_style("white")
        sns.histplot(x = f1, ax=axs[1], bins = 15, edgecolor = '#CCCCCC', facecolor = (rgb_colors[1][0], rgb_colors[1][1], rgb_colors[1][2], 0.25))
        sns.kdeplot(x = f1, ax=axs[1], color = colors[1])
        sns.histplot(x = f2, ax=axs[1], bins = 15, edgecolor = '#CCCCCC', alpha=0.7, facecolor = (rgb_colors[2][0], rgb_colors[2][1], rgb_colors[2][2], 0.25))
        sns.kdeplot(x = f2, ax=axs[1], color = colors[2])

        axs[1].vlines(np.mean(f1), 0, 1, linestyle = 'dashed', transform=axs[1].get_xaxis_transform(), color = colors[1], alpha = 1)
        axs[1].vlines(np.mean(f2), 0, 1, linestyle = 'dashed', transform=axs[1].get_xaxis_transform(), color = colors[2], alpha = 1)
        axs[0].vlines(0.5, 0, 1, linestyle = 'dashed', transform=axs[0].get_xaxis_transform(), color = 'black', alpha = 1)

        axs[1].set(xlabel='Variant Allele Frequency (VAF)', ylabel='Density')
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["left"].set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        axs[1].annotate("Subclone frequency\napproximate\nposteriors", xy=(1,1), xycoords="axes fraction",xytext=(0,-10), textcoords="offset points",ha="right", va="top", fontsize = 8)
        axs[1].set_xlim(0, 1)
        axs[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        
        if save_plot == True:
            fig.savefig(save_dir, dpi=600)

        return axs

    else:
        print('Error. Check predictions input')