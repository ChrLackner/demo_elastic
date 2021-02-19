
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

mp = meshsize.moderate
h_edges = 0.01
mp.RestrictHLine((1.5, 0, 0.2), (1.5, 0.2, 0.2), h_edges)
mp.RestrictH((1.5, 0.8, 0.2), 0.005)

mesh = Mesh(geo.GenerateMesh(mp, maxh=0.07))

Draw(mesh)

E = 210e9
nu = 0.2

load = mesh.BoundaryCF({ "left" : (1, 0, 0),
                         "right" : (-1, 0, 0) }) * 1000

mu = E/2/(1+nu)
lam = E * nu / ((1+nu) * (1-2*nu))

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

I = Id(3)
def Stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * I

fes = VectorH1(mesh, order=3, dirichlet="bottom")
u, v = fes.TnT()
fesStress = H1(mesh, order=2)**9

a = BilinearForm(fes)
a += InnerProduct(Stress(Strain(u)), Strain(v)) * dx

f = LinearForm(fes)
f += load * v * ds("left|right")

a.Assemble()
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

gfustress = GridFunction(fesStress)
gfustress.Set(Stress(Strain(gfu)))

Draw(gfu, mesh, "displacement")
# stress = Stress(Strain(gfu))
stress = gfustress
stress.dims = (3,3)
vonMises = sqrt(1/6 * ((stress[0,0]-stress[1,1])**2 + (stress[1,1] - stress[2,2])**2 + (stress[2,2]-stress[0,0])**2 + 6 * (stress[0,1]**2 + stress[0,2]**2 + stress[1,2]**2)))
Draw(BoundaryFromVolumeCF(Stress(Strain(gfu))), mesh, "stress")
Draw(gfustress, mesh, "averg_stress")
Draw(vonMises, mesh, "von_mises")

from ngsolve.internal import *
visoptions.deformation = 1
visoptions.scaledeform1 = 1e6
visoptions.lineartexture = 1
# visoptions.scalfunction = "averg_stress:9"
