
from netgen.csg import *
from netgen.occ import *
from ngsolve import *
from time import time
from ngsolve.krylovspace import CGSolver, GMRes
from ngsolve.nonlinearsolvers import NewtonSolver
from ngsolve.comp import DifferentialSymbol

simpleGeometry = True
nonlinear = True

thermal_point_source = (0, 0, 0.2)

class MyGMResSolver(BaseMatrix):
    def __init__(self, a, pre):
        super().__init__()
        self.a = a
        self.pre = pre

    def Height(self):
        self.a.mat.height
        
    def Width(self):
        self.a.mat.width

    def Mult(self, x, y):
        y[:] = 0
        GMRes(self.a.mat, x, self.pre, x=y, printrates=True, reltol=1e-4, maxsteps=1000)

if simpleGeometry:
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
        geo.AddPoint(thermal_point_source, "thermal_source")
        return geo
    geo = CreateGeometry()

    mp = meshsize.moderate
    h_edges = 0.01
    # mp.RestrictHLine((1.5, 0, 0.2), (1.5, 0.2, 0.2), h_edges)
    # mp.RestrictH((1.5, 0.8, 0.2), 0.005)

    mesh = Mesh(geo.GenerateMesh(mp, maxh=0.07))
else:
    geo = OCCGeometry("frame.step")
    import os
    meshfile = "frame.vol.gz"

    if os.path.exists(meshfile):
        mesh = Mesh(meshfile)
        mesh.ngmesh.SetGeometry(geo)
    else:
        netgen_mesh = geo.GenerateMesh(meshsize.moderate)
        netgen_mesh.SetBCName(60, "right")
        netgen_mesh.SetBCName(83, "left")
        netgen_mesh.SetBCName(0, "bottom")
        netgen_mesh.SetMaterial(1, "steel")
        netgen_mesh.Save(meshfile)
        mesh = Mesh(netgen_mesh)

    mesh.Curve(4)
Draw(mesh)

E = 210e9
nu = 0.2

if simpleGeometry:
    load = mesh.BoundaryCF({ "left" : (1, 0, 0),
                             "right" : (-1, 0, 0) }) * 1000
else:
    load = mesh.BoundaryCF({ "left" : (0, 0, 1),
                             "right" : (0, 0, -1) }) * 1000


fesT = H1(mesh, order=3, dirichlet="bottom")
uT, vT = fesT.TnT()



aT = BilinearForm(fesT)
aT += 25 * grad(uT) * grad(vT) * dx

dp = DifferentialSymbol(BBBND)
fT = LinearForm(fesT)
fT += 200 * vT * dp("thermal_source")

aT.Assemble()
fT.Assemble()

T = GridFunction(fesT)
T.vec.data = aT.mat.Inverse(fesT.FreeDofs()) * fT.vec

Draw(T, mesh, "T")

input("start elastic...")


mu = E/2/(1+nu)
lam = E * nu / ((1+nu) * (1-2*nu))

if nonlinear:
    I = Id(3)
    def Strain(u):
        F = I + Grad(u)
        return 0.5 * (F.trans * F - I)

    def Stress(strain):
        C = 2 * strain + I
        return mu * (I - Det(C)**(-lam/mu) * Inv(C))

    def Energy(strain):
        C = 2 * strain + I
        return mu * (Trace(strain) + mu/lam * (Det(C)**(-lam/2/mu)-1))

else:
    def Strain(u):
        return 0.5 * (Grad(u) + Grad(u).trans)

    I = Id(3)
    def Stress(strain):
        return 2 * mu * strain + lam * Trace(strain) * I

fes = VectorH1(mesh, order=3, dirichlet="bottom") #, wb_withedges=False)
u, v = fes.TnT()
fesStress = H1(mesh, order=2)**9

start = time()
a = BilinearForm(fes)
if nonlinear:
    a += Variation(Energy(Strain(u)) * dx)
else:
    a += InnerProduct(Stress(Strain(u)), Strain(v)) * dx
print("Biform setup needs", time() - start, "seconds")

pre = Preconditioner(a, type="bddc")

f = LinearForm(fes)
f += load * v * ds("left|right")

@TimeFunction
def Setup():
    a.Assemble()
    f.Assemble()

gfu = GridFunction(fes)

with TaskManager(10**9):
    if nonlinear:
        f.Assemble()
        lin_solver = MyGMResSolver(a, pre)
        newton = NewtonSolver(a, gfu, rhs=f, solver=lin_solver)
        newton.Solve(printing=True, maxerr=1e-8)
    else:
        Setup()
        # gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        # inv = CGSolver(a.mat, pre, tol=1e-8, printing=True, maxsteps=1000)
        # gfu.vec.data = inv * f.vec
        GMRes(a.mat, f.vec, pre, x=gfu.vec,  maxsteps=1000, reltol=1e-8)
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
