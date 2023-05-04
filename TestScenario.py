
from OneDModel import OneDimensionalProplem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gc

"""
Model Geometery
"""
Width = 0.1
Length = 0.01
Height = 0.01
XCells = 100
YCells = 1
ZCells = 1
NumCells = XCells * YCells * ZCells
"""
Material properties
"""
Cs = 2093
Hsf = 333550
Cl = 4180
Tmelt = 273
K = 10
p = 1000
"""
Initial Conditions
"""
Tinitial = 260
"""
Boundary conditions
"""
Tsurr = 260
h = 10
Q = 0.01
"""
Initializing model
"""
Water1 = OneDimensionalProplem(Width, Length, Height, int(XCells/5), Cs, Hsf, Cl, K, p, h, Tmelt, Tinitial, Tsurr, 2, Q, -1)
Water2 = OneDimensionalProplem(Width, Length, Height, XCells, Cs, Hsf, Cl, K, p, h, Tmelt, Tinitial, Tsurr, 2, Q, -1)

PrevTemps1 = np.full((int(XCells/5) * YCells * ZCells), 263)
Temps2 = np.full((XCells * YCells * ZCells), 200)
 
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
Line1, = ax.plot(Water1.Xcoords,PrevTemps1, label='{0} Cells'.format(int(XCells/5)))
Line2, = ax.plot(Water2.Xcoords,Temps2, label='{0} Cells'.format(XCells))
TimeData = [0]
TempData = [Tinitial]
Timestep = 0.1
def Frames():
    Time = 0
    while True:
        yield Time
        Time += Timestep

leg = ax.legend()

ax.set_ylabel("Temperature [K]")

ax.set_xlabel("Distance [m]")
fig.suptitle("Transient temperature response of a two phase problem")

Time = ax.text(0.95, 0.88,"Elapsed time = 0 s", ha="right", va="top", transform=ax.transAxes, bbox=dict(facecolor='lightgreen', alpha=1))

def animationLoop(i):
    
    
    Temps1, x1 = Water1.AdvanceModel(0.1)
    Temps2, x2 = Water2.AdvanceModel(0.1)
    ax.set_ylim(min(min(Temps1),min(Temps2)) - 5, max(max(Temps1),max(Temps2)) + 5)
    minTemp = min(Temps2) - 5
    maxTemp = max(Temps2) + 5
    ax.set_ylim(minTemp, maxTemp)
    Line1.set_ydata(Temps1)
    Line2.set_ydata(Temps2)
    #print("time = {0} s".format(i * Timestep))
    Time.set_text("Elapsed time = {0:.01f} s".format(i))
    TimeData.append(i)
    TempData.append(Temps2[0])

    if (i % 10 == 0): gc.collect()
    return Line1, Line2,

ani = animation.FuncAnimation(fig, animationLoop, frames=Frames, interval=50, save_count=(3600/Timestep))


ax.grid()


f = "animation.mp4"
#writergif = animation.FFMpegFileWriter(fps=20)

#ani.save(f, writer=writergif)
plt.show()
