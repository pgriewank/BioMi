#!/usr/bin/env python
# coding: utf-8

# This is a collection for all the "finished" functions Roel Neggers and Philipp Griewank are using to test their 2D binomial sampling toys. 
# Created 2020-04-03
# The Idea is that all the supporting functions land here, and that all the running and testing happens in the various jupyter notebooks. 


# 



import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from random import random
import matplotlib
from netCDF4 import Dataset
import matplotlib.cm as cm
import matplotlib.pyplot as plt    # The code below assumes this convenient renaming
import math
import seaborn as sns
sns.set()

import sys 
import pandas as pd



#################################################################################################################
# Clustering
#################################################################################################################


def add_buffer(A,n_extra):
    """
    Adds n_extra cells/columns in x and y direction to array A. Works with 2d and 3d arrays, super advanced stuff right here. 
    
    Is used for 2D clustering here. 
    """
    if A.ndim == 2:
        A_extra = np.vstack([A[-n_extra:,:],A,A[:n_extra,:]])
        A_extra = np.hstack([A_extra[:,-n_extra:],A_extra,A_extra[:,:n_extra]])
    if A.ndim == 3:
        A_extra = np.concatenate((A[:,-n_extra:,:],A,A[:,:n_extra,:]),axis=1)
        A_extra = np.concatenate((A_extra[:,:,-n_extra:],A_extra,A_extra[:,:,:n_extra]),axis=2)
        
    
    return A_extra
 


def cluster_2D(A,buffer_size=30 ):
    """
    Not the prettiest version of this script, but should be fine (Philipp) 
    
    
    A is 2D matrix of 1 (cloud) and 0 (no cloud)
    buffer_size is the percentile added to each side to deal with periodic boundary domains
    
    returns labeled_clouds, is 0 where no cloud
    """
    #Uses a default periodic boundary domain
    n_max = A.shape[0]
 
    n_buffer = int(buffer_size/100.*n_max)

    #Explanding c and w fields with a buffer on each edge to take periodic boundaries into account. 
    A_buf=add_buffer(A,n_buffer)
    
    #labeled_clouds  = np.zeros_like(A_buf)
    
    #This is already very impressive, ndi.label detects all areas with marker =1 that are connected and gives each resulting cluster an individual integer value 
    labeled_clouds,n_clouds  = ndi.label(A_buf)
    
    
    
    #Going back from the padded field back to the original size
    # OK, calculate index means, then only look at those with a mean inside the original box
    # We ignore the cells with the mean outside, they will be cut off or overwritten
    # For those inside we check if they have something outside original box, and if so a very ugly hard coded overwritting is done. 
    # In the very end the segmentation box is cut back down to the original size

    
    
    
    #fancy quick sorting. 
    unique_labels, unique_label_counts = np.unique(labeled_clouds,return_counts=True)
    lin_idx       = np.argsort(labeled_clouds.ravel(), kind='mergesort')
    lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(labeled_clouds.ravel())[:-1]))
    
    for c in range(1,n_clouds+1): 
        idx_x,idx_y = np.unravel_index(lin_idx_split[c],labeled_clouds.shape)
    
    

        #idx_x,idx_y = np.where(labeled_clouds==c)
        idx_x_m = np.mean(idx_x)
        idx_y_m = np.mean(idx_y)

        if idx_x_m< n_buffer or idx_x_m>n_buffer+n_max or idx_y_m< n_buffer or idx_y_m>n_buffer+n_max:
            #cluster is outside, chuck it
            #print(c,'cluster out of bounds',idx_x,idx_y)
            #segmentation_cp[segmentation==c] = 0
            bla = 1

        else:
            idx_x_max = np.max(idx_x)
            idx_x_min = np.min(idx_x)
            idx_y_min = np.min(idx_y)
            idx_y_max = np.max(idx_y)
            if idx_x_min< n_buffer or idx_x_max>n_buffer+n_max or idx_y_min< n_buffer or idx_y_max>n_buffer+n_max:
                #print(c,'this is our guniea pig')
                if idx_x_min<n_buffer:
                    idx_x_sel = idx_x[idx_x<n_buffer]+n_max
                    idx_y_sel = idx_y[idx_x<n_buffer]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_x_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_x>=n_buffer+n_max]-n_max
                    idx_y_sel = idx_y[idx_x>=n_buffer+n_max]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_min<n_buffer:
                    idx_x_sel = idx_x[idx_y<n_buffer]
                    idx_y_sel = idx_y[idx_y<n_buffer]+n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_y>=n_buffer+n_max]
                    idx_y_sel = idx_y[idx_y>=n_buffer+n_max]-n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c



    #Now cut to the original domain
    labeled_clouds_orig = labeled_clouds[n_buffer:-n_buffer,n_buffer:-n_buffer]
    
    #And to clean up the missing labels 
    def sort_and_tidy_labels_2D(segmentation):
        """
        For a given 2D integer array sort_and_tidy_labels will renumber the array 
        so no gaps are between the the integer values and replace them beginning with 0 upward. 
        Also, the integer values will be sorted according to their frequency. 
        
        1D example: 
        [4,4,1,4,1,4,4,3,3,3,3,4,4]
        -> 
        [0,0,2,0,2,0,0,1,1,1,1,0,0]
        """
       
        unique_labels, unique_label_counts = np.unique(segmentation,return_counts=True)
        n_labels = len(unique_labels)
        unique_labels_sorted = [x for _,x in sorted(zip(unique_label_counts,unique_labels))][::-1]
        new_labels = np.arange(n_labels)
       
        lin_idx       = np.argsort(segmentation.ravel(), kind='mergesort')
        lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(segmentation.ravel())[:-1]))
        #think that I can now remove lin_idx, as it is an array with the size of the full domain. 
        del(lin_idx)
       
        for l in range(n_labels):
            c = unique_labels[l]
            idx_x,idx_y = np.unravel_index(lin_idx_split[c],segmentation.shape)
            segmentation[idx_x,idx_y] = new_labels[l]
       
        return segmentation 
    

    labeled_clouds_clean = sort_and_tidy_labels_2D(labeled_clouds_orig)
    
    
    return labeled_clouds_clean


#################################################################################################################
# Binning
#################################################################################################################
    




def log_binner_minmax(var, bin_min, bin_max, bin_n, N_min=0):
    """
    written by Lennéa Hayo, 19-07-20
    
    Bins a vector of values into logarithmic bins
    Starting from bin_min and ending at bin_max
    
    Parameters:
        var: input vector
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins 
        
    Returns:
        bins: vector of bin edges, is bin_n+1 long
        ind: gives each value of the input vector the index of its respective bin
        CSD: Non normalized distribution of var over the bins. 
    """
    import numpy as np

    max_val = max(var)
    min_val = min(var)
    #bin_min = max(min_val, bin_min)
    #bin_max = min(max_val, bin_max)

    max_log = np.log10(bin_max / bin_min)

    bins = bin_min * np.logspace(0, max_log, num=bin_n + 1)
    ind = np.digitize(var, bins)
    CSD = np.zeros(bin_n)
    for b in range(bin_n):
        if len(ind[ind == b + 1]) > N_min:
            CSD[b] = float(np.count_nonzero(ind == b + 1)) / (bins[b + 1] - bins[b])
        else:
            CSD[b] = 'nan'
    return bins, ind, CSD 


# In[7]:


def lin_binner_minmax(var, bin_min, bin_max, bin_n, N_min=0):
    """
    written by Philipp Griewank 20-01-21
    
    Bins a vector of values into linear bins
    Starting from bin_min and ending at bin_max
    
    Parameters:
        var: input vector
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins 
        
    Returns:
        bins: vector of bin edges, is bin_n+1 long
        ind: gives each value of the input vector the index of its respective bin
        CSD: Non normalized distribution of var over the bins. 
    """
    import numpy as np

    max_val = max(var)
    min_val = min(var)
    #bin_min = max(min_val, bin_min)
    #bin_max = min(max_val, bin_max)


    bins = np.linspace(bin_min, bin_max, num=bin_n + 1)
    ind = np.digitize(var, bins)
    CSD = np.zeros(bin_n)
    for b in range(bin_n):
        if len(ind[ind == b + 1]) > N_min:
            CSD[b] = float(np.count_nonzero(ind == b + 1)) / (bins[b + 1] - bins[b])
        else:
            CSD[b] = 'nan'
    return bins, ind, CSD 


#################################################################################################################
# Plotting 
#################################################################################################################

def func_generate_random_cmap(N_color = 1000):
    """
    Makes a random colormap with N_color individual colors
    """
    colors = [(0.5,0.5,0.5)] + [(random(),random(),random()) for i in range(N_color)]
    random_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('random_cmap', colors, N=N_color)
    return random_cmap


def func_scatter_grid(N_2D,buffer=0.1):
    """
    Just used for fancy plotting. 
    
    Takes the 2D N array which has the number of objects per cell and distributes them randomly within them in a 2D x-y plane. Unit is per gridbox, beware. 
    Buffer is how much space is left around the end
    """

    x_rand_grid = np.zeros(0)
    y_rand_grid = np.zeros(0)
    n_width,n_height = N_2D.shape
    for i in range(n_width):
        for j in range(n_height):
            x_rand_grid = np.hstack([x_rand_grid,np.zeros(N_2D[i,j])+i])
            y_rand_grid = np.hstack([y_rand_grid,np.zeros(N_2D[i,j])+j])

    x_rand_grid = x_rand_grid + np.random.random(len(x_rand_grid))*(1-2*buffer)+buffer
    y_rand_grid = y_rand_grid + np.random.random(len(y_rand_grid))*(1-2*buffer)+buffer
    return x_rand_grid,y_rand_grid    


#def func_advection_numerics_plot(N1,N2,N3,label1=r'$\Delta t$: 12 min',label2=r'$\Delta t$: 6 min',label3=r'$\Delta t$: 3 min'):
def func_advection_numerics_plot(N1,N2,N3,label1='$\Delta t$: 12 min',label2='$\Delta t$: 6 min',label3='$\Delta t$: 3 min'):
    """
    Plots the final timestep and the time evolution of the mean profile of 3 experiments. 
    
    Highly specific to the advection/diffusion plot, and needs lots of input variables not defined above. Is stored in the general function py file to serve as a reference for other plot scripts. 
    
    Created by Philipp Griewank 2020-03 
    
    """
    
    #settings
    sizes       = [20,100]
    colors      = ['b','orange']
    edge_colors = ['b','r']
    color_mean  = ['k','w']
    alphas      = [0.3,1]
    edge_dist   = [0.85,0.75]

    
    from matplotlib.patches import Rectangle
    
    
    fig,axes = plt.subplots(1,3,figsize=(4*3,4),sharex=True,sharey=True)
    
    
    axes[0].add_patch(Rectangle((0,0),1,1,edgecolor='k',fill=True,color='r',lw=2,ls='--',alpha=0.3,zorder=0))
    axes[1].add_patch(Rectangle((0,0),1,1,edgecolor='k',fill=True,color='r',lw=2,ls='--',alpha=0.3,zorder=0))
    axes[2].add_patch(Rectangle((0,0),1,1,edgecolor='k',fill=True,color='r',lw=2,ls='--',alpha=0.3,zorder=0))

    #first the scatter plots
    for l in range(2):
        x, y = func_scatter_grid(N1[:,:,l,-1],buffer=edge_dist[l])
        axes[0].scatter(x,y,s=sizes[l],c=colors[l],alpha=alphas[l],edgecolors=edge_colors[l])
        
        x, y = func_scatter_grid(N2[:,:,l,-1],buffer=edge_dist[l])
        axes[1].scatter(x,y,s=sizes[l],c=colors[l],alpha=alphas[l],edgecolors=edge_colors[l])
        
        x, y = func_scatter_grid(N3[:,:,l,-1],buffer=edge_dist[l])
        axes[2].scatter(x,y,s=sizes[l],c=colors[l],alpha=alphas[l],edgecolors=edge_colors[l])

    #Now the mean values of each timestep
    #First setting up the meshgrid
    n_x = np.shape(N1)[0]
    n_y = np.shape(N1)[1]
    y_mesh, x_mesh = np.meshgrid(np.arange(n_x)+0.5,np.arange(n_y)+0.5)
    
    N  = N1
    ax = axes[0] 
    
    n_time = np.shape(N)[3]
    x_mean = np.zeros([n_time,2])
    y_mean = np.zeros([n_time,2])
    for l in [1,0]:
        for t in range(n_time):
            x_mean[t,l] = np.sum(np.multiply(x_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
            y_mean[t,l] = np.sum(np.multiply(y_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
    for l in [1,0]:
        ax.plot(x_mean[:,l],y_mean[:,l],'k')
        ax.scatter(x_mean[:,l],y_mean[:,l],s=sizes[l],color=color_mean[l],edgecolors='k')
    
    N  = N2
    ax = axes[1] 
    
    n_time = np.shape(N)[3]
    x_mean = np.zeros([n_time,2])
    y_mean = np.zeros([n_time,2])
    for l in [1,0]:
        for t in range(n_time):
            x_mean[t,l] = np.sum(np.multiply(x_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
            y_mean[t,l] = np.sum(np.multiply(y_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
    for l in [1,0]:
        ax.plot(x_mean[:,l],y_mean[:,l],'k')
        ax.scatter(x_mean[:,l],y_mean[:,l],s=sizes[l],color=color_mean[l],edgecolors='k')
    
    N  = N3
    ax = axes[2] 
    
    n_time = np.shape(N)[3]
    x_mean = np.zeros([n_time,2])
    y_mean = np.zeros([n_time,2])
    for l in [1,0]:
        for t in range(n_time):
            x_mean[t,l] = np.sum(np.multiply(x_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
            y_mean[t,l] = np.sum(np.multiply(y_mesh,N[:,:,l,t]))/np.sum(N[:,:,l,t])
    for l in [1,0]:
        ax.plot(x_mean[:,l],y_mean[:,l],'k')
        ax.scatter(x_mean[:,l],y_mean[:,l],s=sizes[l],color=color_mean[l],edgecolors='k')
    
    axes[0].grid(True,color='k',lw=1,alpha=0.5)
    axes[1].grid(True,color='k',lw=1,alpha=0.5)
    axes[2].grid(True,color='k',lw=1,alpha=0.5)

    axes[0].set_xlabel('x [km]')
    axes[1].set_xlabel('x [km]')
    axes[2].set_xlabel('x [km]')
    axes[0].set_title(label1)
    axes[1].set_title(label2)
    axes[2].set_title(label3)

    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.xlim(0,n_x)
    plt.ylim(0,n_y)
    
    axes[0].set_ylabel('y [km]')
    return fig


def func_scatter_grid_time(N_2D,buffer=0.1,t=1,t_substeps=10.,t_juggle=True):
    """
    The idea is to have something similar to func_scatter_grid, but with smaller movements between calls so that a video can be made where things don't go crazy
    
    Next attempt, have the individual objects get a new seed every x timesteps, and move from one to the other. 
    
    Problems, they all start moving to a new destination at the same time, looks super choreographed
    Needs things to be different for the individual size bins
    
    Buffer is how much space is left at the edge
    """
    
    # Original version that uses a fixed t, so that all grids change direction at the same time
    t_0 = np.floor(t/t_substeps)
    t_ratio = (t-t_0*t_substeps)/t_substeps
    t_1 = t_0+1
    
    offset_y = int(1e6) #offset for y seed

    x_rand_grid = np.zeros(0)
    y_rand_grid = np.zeros(0)
    n_width,n_height = N_2D.shape
    for i in range(n_width):
        for j in range(n_height):
            
            if t_juggle:# Chaos version! Grid cells have a slightly modified t
                t_pert = t + i*j
                t_0 = np.floor(t_pert/t_substeps)
                t_ratio = (t_pert-t_0*t_substeps)/t_substeps
                t_1 = t_0+1
            
            np.random.seed(i*n_width+j+int(t_0)*n_width*n_height)
            local_rand_x_0 = (np.random.random(N_2D[i,j])-0.5)*(1-2*buffer)
            np.random.seed(i*n_width+j+int(t_0)*n_width*n_height+offset_y)
            local_rand_y_0 = (np.random.random(N_2D[i,j])-0.5)*(1-2*buffer)
            
            np.random.seed(i*n_width+j+int(t_1)*n_width*n_height)
            local_rand_x_1 = (np.random.random(N_2D[i,j])-0.5)*(1-2*buffer)
            np.random.seed(i*n_width+j+int(t_1)*n_width*n_height+offset_y)
            local_rand_y_1 = (np.random.random(N_2D[i,j])-0.5)*(1-2*buffer)
            
            
            
            
            local_rand_x = (1-t_ratio)*local_rand_x_0 + t_ratio*local_rand_x_1
            local_rand_y = (1-t_ratio)*local_rand_y_0 + t_ratio*local_rand_y_1 

            
            x_rand_grid = np.hstack([x_rand_grid,local_rand_x+i+0.5])
            y_rand_grid = np.hstack([y_rand_grid,local_rand_y+j+0.5])

    return x_rand_grid,y_rand_grid    

def plot_scatter_time(N,t,
    sizes  = [1,10,20,50,90],
    color  = ['k','b','orange','r','g'],
    buffer = [0.9,0.85,0.8,0.75,0.7],
    alpha  = [0.5,0.5,0.5,0.5,0.5],
    labels = ['','','','',''],
    figsize = (6,6),
    time_tracking=False,
    t_substeps=10,
    dx=1,dy=1,
    rotation = 0 ):
    """
    A simple single snapshot scatter plot with random displacement
    
    Importanly, if a movie is being made time_tracking needs to be set to True
    """


    from matplotlib.ticker import MultipleLocator
   
    
    fig = plt.figure(figsize=figsize)
    axes = plt.gca()
    
    nx,ny,nsiz,ntim,nage= N.shape
     
    #axes.set_aspect('equal')
    axes.set_xlim([0,nx*dx])
    axes.set_ylim([0,ny*dy])
    
    axes.xaxis.set_major_locator(MultipleLocator(dx))
    axes.yaxis.set_major_locator(MultipleLocator(dy))   #set gridspacing
    
    for l in range(nsiz):
        
        #Just add with l*1000! Philipp you are so smart
        #checking if the 
        if time_tracking:
            xs, ys = func_scatter_grid_time( np.sum(N[:,:,l,t,:],axis=2),t=t+l*1000+l,t_substeps=t_substeps,buffer=buffer[l])
        else:
            xs, ys = func_scatter_grid( np.sum(N[:,:,l,t,:],axis=2),buffer=buffer[l])
        axes.scatter(dx*xs, dy*ys, s=sizes[l],c=color[l],alpha=alpha[l],label=labels[l])
    
    plt.title("timestep: %s" % (t) )
    plt.xlabel("%s  [%s]" % ("x","m"))
    plt.ylabel("%s  [%s]" % ("y","m"))
    plt.xticks(rotation=rotation)
    
    return fig


#################################################################################################################
# Movies
#################################################################################################################

def plot_scatter_movie(N,t0,t1,output_folder='mov/',base_str='N_movie_',name='',dpi=200,
    sizes         = [1,10,20,50,90],
    color         = ['k','b','orange','r','g'],
    buffer        = [0.9,0.85,0.8,0.75,0.7],
    alpha         = [0.5,0.5,0.5,0.5,0.5],
    figsize       = (6,6),
    time_tracking = True,
    t_substeps    = 10,
    dx=1,dy=1,
    rotation=0):
    """
    A wrapper for plot_scatter_time that loops from t0 to t1 plotting the scatter plots for each and saving them.
    
    Currently only works for up to 5 different objects
    WARNNG: Seems to have a memory leak! Don't run forever.
    """
    
    for t in range(t0,t1): 
        fig = plot_scatter_time(N,t,
        sizes         = sizes     ,
        color         = color     ,
        buffer        = buffer    ,
        alpha         = alpha     ,
        figsize       = figsize   ,
        time_tracking = time_tracking,
        t_substeps    = t_substeps,
        dx = dx, dy =dy,
        rotation = rotation)
        
        ext = ".png"
        savestring =base_str+name+'_dt'+str(t)+ext
        if t<100:
            savestring =base_str+name+'_dt0'+str(t)+ext
        if t<10:
            savestring =base_str+name+'_dt00'+str(t)+ext

        fig.savefig(output_folder+savestring,bbox_inches = 'tight',dpi=200)
        print('saved: '+output_folder+savestring)
        plt.close(fig)
    
    return


#################################################################################################################
# Advection 
#################################################################################################################

def func_advection_binomial_2D(N_sub_2D_t0,v_x_rel,v_y_rel,periodic_flag = True):
    """
    New and improved version of the binomial advection funtion for 2D rectangular grids. 
    
    created by Philipp Griewakn 2020-03-24
    
    General approach of the function is that it first does the fractional advection (i.e. advection that is not a full gridbox) using binomial trials. Afterwards it moves the objects around the full grid spacing 
    For example:
    for v_x_rel = 0.5, everything is moved half a grid to the right using binomial sampling
    for v_x_rel =-0.5, everything is moved half a grid to the right using binomial sampling, then shifted one to the left afterwards
    for v_x_rel = 1.5, everything is moved half a grid to the right using binomial sampling, then shifted one to the right afterwards
    
    The binomial trials first go in x, than in y, and then in xy directions. probabilities p are calculated accordingly.
    Does not work if the advections is farther than the total grid size, i.e. around the 2D grid in one timestep. 
    Should be pretty dirt cheap. If not, Philipp coded something shitty.  
    
    Includes periodic boundary conditions by default, and automaticall checks for conservation if enabled
   
    Parameters
    -----------
    N_sub_2D_t0   : 2D array of the number of objects on the 2D grid
    v_x_rel       : Advection speed devided by grid spacing in x direction     
    v_y_rel       : Advection speed devided by grid spacing in y direction     
    periodic_flag : Set to True by default, adds a periodic boundary to the 2D domain. If set to False, advection from outside the domain is set to zero
    
    Returns
    -----------
    N_sub_2D_t1   : New 2D array of where the objects are post advection.  
    
    Examples
    -----------
    N = np.zeros([10,10]).astype(int)
    N[0:2,0:2] = 10
    N = func_advection_binomial_2D(N,2.5,-3.2)
    plt.pcolormesh(N) 
    """
   
    # first thing is to split the relative displacement into fractional and whole displacement
    # The fractional displacement is done first, and is equivalent to the original displacement that only allowed movements of up to one grid cell
    # For the meantime I will try to formulate things so they are fine with negative v_x and v_y
    
    # After that everything is moved whole gridboxes in a direction for the whole displacement 
    
    v_x_rel_whole = int(np.floor(v_x_rel))  
    v_y_rel_whole = int(np.floor(v_y_rel))
    
    v_x_rel_frac = v_x_rel-v_x_rel_whole  
    v_y_rel_frac = v_y_rel-v_y_rel_whole
    
    #This only works as long as the whole displacement is smaller than the total size of the domain, so:
    if abs(v_x_rel_whole)>=N_sub_2D_t0.shape[0]:
        print  ('x displacement is greater than grid size, advection routine does not like this')
        return -1
    if abs(v_y_rel_whole)>=N_sub_2D_t0.shape[1]:
        print  ('y displacement is greater than grid size, advection routine does not like this')
        return -1
    
    #So, fractional displacement starts here: 
    
    #The Areas and probabilities should are calculated from the fractional advection values, this wasn't really carefully thought through, but I think it should be correct
    
    A_x  = v_x_rel_frac-v_x_rel_frac*v_y_rel_frac
    A_y  = v_y_rel_frac-v_x_rel_frac*v_y_rel_frac
    A_xy = v_y_rel_frac*v_x_rel_frac
    A = 1
    p_x  = A_x/A
    p_y  = A_y/(A-A_x)
    p_xy = A_xy/(A-A_x-A_y)
    
    #Now do and x, y, and xy separately, and loop over all N_sub
    N_x  = np.zeros_like(N_sub_2D_t0)
    N_y  = np.zeros_like(N_sub_2D_t0)
    N_xy = np.zeros_like(N_sub_2D_t0)
    
    #Binomal sampling in the 3 directions 
    N_x  = np.random.binomial(N_sub_2D_t0,p_x)
    N_y  = np.random.binomial(N_sub_2D_t0-N_x,p_y)
    N_xy = np.random.binomial(N_sub_2D_t0-N_x-N_y,p_xy)
    
    #Applying the binomial advection to the initial state, including periodic boudnary domain
    N_sub_2D_t1 = N_sub_2D_t0 +0
    
    N_sub_2D_t1[1: ,:] = N_sub_2D_t1[1: ,:] + N_x[:-1,:] 
    N_sub_2D_t1[:,1: ] = N_sub_2D_t1[:,1: ] + N_y[:,:-1] 
    N_sub_2D_t1[1: ,1: ] = N_sub_2D_t1[1: ,1: ] + N_xy[:-1,:-1] 
    
    
    N_sub_2D_t1[:,:] = N_sub_2D_t1[:,:] - N_x[:,:] 
    N_sub_2D_t1[:,:] = N_sub_2D_t1[:,:] - N_y[:,:] 
    N_sub_2D_t1[:,:] = N_sub_2D_t1[:,:] - N_xy[:,:] 
    if periodic_flag==True: 

        #periodic boundaries in x and y is easy
        N_sub_2D_t1[0 ,:] = N_sub_2D_t1[0 ,:] + N_x[-1,:] 
        N_sub_2D_t1[:,0 ] = N_sub_2D_t1[:,0 ] + N_y[:,-1]

        #for xy is a pain in the ass. Needs the first row, first column, and 0,0 thing separately
        N_sub_2D_t1[0  ,1: ] = N_sub_2D_t1[0 ,1: ] + N_xy[-1,:-1] 
        N_sub_2D_t1[1: , 0 ] = N_sub_2D_t1[1: ,0 ] + N_xy[:-1,-1] 
        N_sub_2D_t1[0  , 0 ] = N_sub_2D_t1[0 ,0 ] + N_xy[-1,-1] 

    
    # Now the whole displacement, which is a simple movement in x and y direction. Just have to sort out how to do write it nicely and deal with periodic boundary domains.  
    # Need to make a copy of the N_sub_2D_t1 to not overwrite stuff
    N_sub_copy = N_sub_2D_t1 + 0.
    
    # x direction
    t = v_x_rel_whole
    if t > 0:
        N_sub_2D_t1[t:,:]=N_sub_copy[:-t,:]
        if periodic_flag:
            N_sub_2D_t1[:t,:]=N_sub_copy[-t:,:]
        else:       
            N_sub_2D_t1[:t,:] = 0.
    if t < 0:
        N_sub_2D_t1[:t,:]=N_sub_copy[-t:,:]
        if periodic_flag:
            N_sub_2D_t1[t:,:]=N_sub_copy[:-t,:]
        else:      
            N_sub_2D_t1[t:,:] = 0.
    
    N_sub_copy = N_sub_2D_t1 + 0.
    
    # y direction
    t = v_y_rel_whole
    if t > 0:
        N_sub_2D_t1[:,t:]=N_sub_copy[:,:-t]
        if periodic_flag:
            N_sub_2D_t1[:,:t]=N_sub_copy[:,-t:]
        else:       
            N_sub_2D_t1[:,:t] = 0.
    if t < 0:
        N_sub_2D_t1[:,:t]=N_sub_copy[:,-t:]
        if periodic_flag:
            N_sub_2D_t1[:,t:]=N_sub_copy[:,:-t]
        else:        
            N_sub_2D_t1[:,t:] = 0.
    
    if periodic_flag:
        if np.sum(N_sub_2D_t0) != np.sum(N_sub_2D_t1):
            print('wtf! No advection conservation')
    
    
    
    return N_sub_2D_t1



#################################################################################################################
# timestepping
#################################################################################################################
# These functions are the engines of the various simulations, and need a wide range of stuff
# To be declared beforehand. 
# In general I assume that the idividual notebooks will contain their own various timestepping routines
# This is included here more as a reference. 
#################################################################################################################



def func_timestepping_simple():
    """
    Simplest version of timestepping which only has binomial birth and discrete death through demporaphics.  
    
    """

    #--- time loop ----
    for t in range(ntim):


        #--- size loop ---
        for l in range(nsiz):

            
            #--- Object birthdays! Time-shift of the demographics levels ---
            for i in range(nage-1,0,-1):
                N[:,:,l,t,i] = N[:,:,l,t-1,i-1]   #note: oldest level is forgotten
            
            

            #--- Births ---
            B = np.random.binomial(N_tot_ref[l], p[:,:,l,t].ravel(), nx*ny)
            B = B.reshape([nx,ny])

            
            #--- Update object number ---
            N[:,:,l,t,0] = B    # add births as level 1 demographics
            
            
            print( "t=%s  time = %s" % (t, t*dtim),' sizebin' ,l,'max birth: ',np.max(B),np.max(N[:,:,l,t,0]))

        
    return N, p





