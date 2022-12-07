import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animateTSP(history, xx, yy, costs):
    ''' animate the solution over time

        Parameters
        ----------
        hisotry : list
            history of the solutions chosen by the algorith
        points: array_like
            points with the coordinates
    '''

    ''' approx 1500 frames for animation '''
    points = np.column_stack((xx, yy))

    div = len(history) // 3


    key_frames_mult = len(history) // div

    fig, ax = plt.subplots()

    title = ax.text(0.8,1.035, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
    ''' path is a line coming through all the nodes '''
    line, = plt.plot([], [], lw=2)

    def init():
        ''' initialize node dots on graph '''
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        plt.plot(x, y, 'co')

        ''' draw axes slighty bigger  '''
        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

        '''initialize solution to be empty '''
        line.set_data([], [])
        return line,

    def update(frame):
        ''' for every frame update the solution on the graph '''
        x = [points[i, 0] for i in history[frame] + [history[frame][0]]]
        y = [points[i, 1] for i in history[frame] + [history[frame][0]]]
        title.set_text("Iteracion {}, costo {}".format(frame, costs[frame]))
        line.set_data(x, y)
        return line

    ''' animate precalulated solutions '''

    ani = FuncAnimation(fig, update, frames=range(0, len(history), key_frames_mult), init_func=init, interval=3, repeat=False)
    plt.title("Ruta - TSP")
    plt.show()
