
from netgen.csg import *
from ngsolve import *

def CreateGeometry():
    geo = CSGeometry()
    bottom = Plane((0,0,0), (0,0,-1)).bc("bottom")
    plate_larger = OrthoBrick((0,0,-1), (2, 1, 0.2))
    plate = plate_larger * bottom
    top = Plane((0,0,1), (0,0,1))
    left = OrthoBrick((1.5, 0, 0.2), (1.7, 0.2, 2)) * top
    right = OrthoBrick((1.5, 0.8, 0.2), (1.7, 1, 2)) * top
    geo.Add(plate.mat("steel"))
    geo.Add(left.mat("steel"), bcmod=[(top, "left")])
    geo.Add(right.mat("steel"), bcmod=[(top, "right")])
    return geo

geo = CreateGeometry()
mesh = Mesh(geo.GenerateMesh())

Draw(mesh)


