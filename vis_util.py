import tkinter
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import random
import matplotlib.colors as colors
matplotlib.use('TkAgg')


total_num = 0

colors_list = list(colors._colors_full_map.values())
random.shuffle(colors_list)
colors_list = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"] + colors_list
print(colors_list[100])

print("number of colours", len(colors_list))


def show_spheres(weights=[], id=0, centres=[], show=False, savefile=""):
    n = len(weights)
    figure, axes = plt.subplots()
    clist=[]
    x0, x1, y0, y1 = 100, 0, 100, 0
    for i in range(n):
        O, R = weights[i][:-1], math.exp(weights[i][-1])
        if x0 > O[0] - R: x0 = O[0] - R
        if x1 < O[0] + R: x1 = O[0] + R
        if y0 > O[1] - R: y0 = O[1] - R
        if y1 < O[1] + R: y1 = O[1] + R

        if i == 0:
            draw_circle = plt.Circle(O, R, color=colors_list[i % len(colors_list)], fill=False, label='S')
        elif i == n-1:
            draw_circle = plt.Circle(O, R, color=colors_list[i % len(colors_list)], fill=False,linestyle=(0,(1,1)), label='P')
        else:
            draw_circle = plt.Circle(O, R, color=colors_list[i % len(colors_list)], fill=False, linestyle=(0,(5,5)), label='M{}'.format(i))

        axes.set_aspect(1)
        axes.add_artist(draw_circle)
        clist.append(draw_circle)
    figure.legend(handles=clist, loc="upper right")

    for c in centres:
        if len(weights) > 0:
            O, R = c, weights[id][-1]
        else:
            O, R = c, 1
        draw_circle = plt.Circle(O, math.exp(R), color='black', fill=False)
        axes.set_aspect(1)
        axes.add_artist(draw_circle)

    plt.xlim(x0-0.1, x1+0.1)
    plt.ylim(y0-0.1, y1+0.1)
    if show:
        plt.show()
    if len(savefile) > 0:
        figure.savefig(savefile)


def visualizing_spatial_layout(sphereLst, fn):
    """
    :param sphereLst: [[(0,0), 1, 'John'], [(2,2), 1, 'Mary']]
    :param spatial_idx:
    :return:
    """
    minX, maxX, minY, maxY, maxR = 0, 10, 0, 10, 1
    cLst, i = dict(), 0
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.legend(["SpatioID"], [str(spatial_idx)])
    i = 0
    for s in sphereLst:
        if s[1] == 0: continue
        cLst[i] = plt.Circle(s[0], s[1], color=colours[i % num_colours], fill=False)
        print(s)
        plt.text(s[0][0]-0.6, s[0][1], s[2])
        ax.add_artist(cLst[i])
        # ax.legend([cLst[i]], [s[2]])
        x, y, r = s[0][0], s[0][1], s[1]
        if minX > x: minX = x
        if maxX < x: maxX = x
        if minY > y: minY = y
        if maxY < y: maxY = y
        if maxR < r: maxR = r
        i += 1
    plt.xlim(minX -maxR, maxX +maxR)
    plt.ylim(minY -maxR, maxY +maxR)
    plt.savefig(fn)
    plt.clf()
    # plt.show()


    """

    cLst[0] = plt.Circle((0, 0), 1, color=colours[i % num_colours], fill=False)
    cLst[1] = plt.Circle((2, 2), 1, color=colours[i % num_colours], fill=False)
    ax.add_artist(cLst[0])
    ax.add_artist(cLst[1])
    """

if __name__ == '__main__':
    show_spheres(weights=[[0,0,1], [1,2,3]], id=0, centres=[], show=True, savefile="test_fig")
    #visualizing_spatial_layout([[(2,2), 1, "mary"], [(0,0), 1, "john"]], spatial_idx=-1)