import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import os

def hist_plot(values, name):
    num_bins = 100
    n, bins, patches = plt.hist(values,num_bins , facecolor='blue', alpha=0.5)
    #n, bins, patches = plt.hist(values, facecolor='blue', alpha=0.5)
    plt.title("{} distribution".format(name))
    plt.savefig("distribution_{}.png".format(name))
    plt.close()

def plot(var1, nameVar1, var2, nameVar2):
    plt.plot( var1, label = nameVar1)
    plt.plot( var2, label = nameVar2)
    plt.legend()
    plt.savefig( nameVar1+"_"+nameVar2+ ".png")
    plt.close()

def plot(var1, nameVar1):
    plt.plot( var1, label = nameVar1)
    
    plt.legend()
    plt.savefig(nameVar1+".png")
    plt.close()

def plotXY(x, y, name):
    plt.plot(x, y)
    plt.xlabel("Distance from wall [cm]")
    plt.ylabel("B")
    plt.savefig(name+".png")
    plt.close()

def kde_plot(values, name, path=None):
    sns.kdeplot(values, fill = True , color = "Blue")
    #plt.xlim([-100, 100])
    plt.title("{} distribution".format(name))
    if path!= None:
        plt.savefig(os.path.join(path,"distribution_{}.png".format(name)))
    else:
        plt.savefig("distribution_{}.png".format(name))
    plt.close()


def list_kde_plot(values, name, logscale=False):

    #fig, ax = plt.subplots(figsize=(10, 6))
    colors = [ "Red", "Blue","Orange", "Purple", "Green", "Yellow"]
    i = 0
    for key, val in values.items():
        sns.kdeplot(val, fill = True , color = colors[i], label=key)
        i += 1
       
    plt.title("{} distribution".format(name))
    plt.legend()
    #plt.tight_layout()
    if logscale:
        plt.xscale("log")
    plt.savefig("distribution_{}.png".format(name))
    plt.close()

def plotSpatialDistribution(coord, errors, error_threshold, name=None):

    # Error threshold for plotting
    msk_error = (np.absolute(errors) > error_threshold) 

#    print(f" Mask shape {msk_error.shape}")
#    print(f" Coord shape {coord.shape}")
    cut_coord = [ coord[:,i] for i in range(msk_error.size) if msk_error[i] ]
    cut_coord = np.array(cut_coord)  #coord[:, msk_error]
#    print(f" Cut coord shape {cut_coord.shape}")
    msk_error_MAP = errors[msk_error] # np.absolute(errors[msk_error])


     #axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("flare").as_hex())

    # plot
    sc = ax.scatter(cut_coord[:,0], cut_coord[:,1], cut_coord[:,2], s=20, marker='o', c = msk_error_MAP, cmap=cmap, alpha=1)
    clb = fig.colorbar(sc, ax = ax, shrink = 0.5, pad = 0.1)
    clb.ax.set_title('Error [%]')

    ax.set_xlim( np.min(coord[0,:]), np.max(coord[0,:]))
    ax.set_ylim( np.min(coord[1,:]), np.max(coord[1,:]))
    ax.set_zlim( np.min(coord[2,:]), np.max(coord[2,:]))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Distribution of errors > {error_threshold} %")


    plt.savefig("{}_spatials_errors_distrib_THRESHOLD_{}.png".format(name, error_threshold))
    plt.close()

def Plot2D(PredDose, RealDose, name):

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(30, 20)

    plt.subplot(1,2,1)
    plt.title("Predicted Dose distribution", fontsize=18)
    plt.imshow(PredDose[:, 0, :], cmap='jet', vmin = RealDose.min().item(), vmax = RealDose.max().item(), origin='lower')
    plt.colorbar()


    plt.subplot(1,2,2)
    plt.title("Real Dose distribution", fontsize=18)
    plt.imshow(RealDose[:, 0, :], cmap='jet', vmin = RealDose.min().item(), vmax = RealDose.max().item(), origin='lower')
    plt.colorbar()


    plt.savefig("{}_2D_dose_distribution.png".format(name))
    plt.close()
