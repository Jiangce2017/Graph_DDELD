import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

def plot_biggraph(points, edge_index,figure_path):
    fig = plt.figure()
    ax = fig.add_subplot()

    p1s = points[edge_index[0, :]]
    p2s = points[edge_index[1, :]]
    lines = np.concatenate((p1s[:,np.newaxis,:],p2s[:,np.newaxis,:]),axis=1)
    line_collection = LineCollection(lines,colors='blue',linewidths=0.1)
    ax.add_collection(line_collection)

    ax.plot(points[:, 0], points[:, 1], '.b', markersize=1)
    ax.set_aspect('equal', 'box')
    plt.savefig(figure_path)

def plot_subgrpahs(c_idx,points, edge_index,sub_points,sub_edge_index,figure_path):
    fig = plt.figure()
    ax = fig.add_subplot()
    p1s = points[edge_index[0, :]]
    p2s = points[edge_index[1, :]]
    lines = np.concatenate((p1s[:,np.newaxis,:],p2s[:,np.newaxis,:]),axis=1)
    line_collection = LineCollection(lines,colors='blue',linewidths=0.1)
    ax.add_collection(line_collection)
    ax.plot(points[:, 0], points[:, 1], '.b', markersize=1)
    
    sub_p1s = sub_points[sub_edge_index[0,:]]
    sub_p2s = sub_points[sub_edge_index[1,:]]
    sub_lines = np.concatenate( (sub_p1s[:,np.newaxis,:],sub_p2s[:,np.newaxis,:]),axis=1)
    sub_line_collection = LineCollection(sub_lines,colors='red',linewidths=0.2)
    ax.add_collection(sub_line_collection)
    ax.plot(sub_points[:, 0], sub_points[:, 1], '.r', markersize=1.5)


    ax.plot(sub_points[c_idx,0],sub_points[c_idx,1],'go',markersize=3)
    ax.set_aspect('equal', 'box')

    plt.savefig(figure_path)

def plot_3Dgraph(points, edge_index,boundary_mask,figure_path):
        points = points.numpy()
        edge_index = edge_index.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p1s = points[edge_index[0, :]]
        #ax.plot(p1s[:, 0], p1s[:, 1], p1s[:, 2], '.r', markersize=1)

        p2s = points[edge_index[1, :]]
        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors='b')       
        #ax.add_collection(lc)
        inside_points = points[~boundary_mask]
        ax.plot(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], '.r', markersize=1)
        bound_points = points[boundary_mask]
        ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], '.g', markersize=1)
        ax.set_aspect('equal', 'box')
        plt.savefig(figure_path)
        #plt.show()