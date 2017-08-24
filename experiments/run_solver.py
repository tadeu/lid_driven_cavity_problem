import logging
import sys
import time

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper, scipy_solver_wrapper
from lid_driven_cavity_problem.residual_function import numpy_residual_function, \
    pure_python_residual_function, cython_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    run_solver(
        solver_type='petsc',
        kernel_type='numpy',
        plot_results=True,
    )


def run_solver(
    solver_type,
    kernel_type,
    nx=40,
    ny=40,
    plot_results=False,
    ):
    if solver_type == 'petsc':
        solver = petsc_solver_wrapper.solve
    elif solver_type == 'scipy':
        solver = scipy_solver_wrapper.solve
    else:
        print("WARNING: Unknown solver type %s. Will use default solver." % (solver_type,))
        solver = None

    if kernel_type == 'python':
        residual_f = pure_python_residual_function.residual_function
    elif kernel_type == 'numpy':
        residual_f = numpy_residual_function.residual_function
    elif kernel_type == 'cython':
        residual_f = cython_residual_function.residual_function
    else:
        print("WARNING: Unknown residual function %s. Will use default." % (solver_type,))
        solver = None

    size_x = 1.0
    size_y = 1.0
    dt = 1e-2
    rho = 1.0
    final_time = 200.0
    mi = 1.0
    Re = 10.0
    U_bc = (mi * Re) / (rho * size_x)
    print("Run Parameters:")
    print("size_x = %s" % (size_x,))
    print("size_y = %s" % (size_y,))
    print("nx = %s" % (nx,))
    print("ny = %s" % (ny,))
    print("dt = %s" % (dt,))
    print("rho = %s" % (rho,))
    print("mi = %s" % (mi,))
    print("U_bc = %s" % (U_bc,))
    print("Re = %s" % (Re,))
    print("")

    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
    b = time.time()
    result = run_simulation(graph, final_time, solver, residual_f)
    print(time.time() - b)

    if plot_results:

        U = np.array(result.ns_x_mesh.phi)
        V = np.array(result.ns_y_mesh.phi)

        U = U.reshape(nx, ny - 1)
        V = V.reshape(nx - 1, ny)

        U = np.c_[[0.0] * nx, U, [0.0] * ny]
        U = (U[:, 1:] + U[:, :-1]) / 2.0

        V = np.r_[[[0.0] * nx], V, [[0.0] * ny]]
        V = (V[1:, :] + V[:-1, :]) / 2.0

        X, Y = np.meshgrid(np.arange(0.0, size_x, size_x / nx), np.arange(0.0, size_y, size_y / ny))
        plt.figure(1)
        plt.title("Velocity streamlines")
        plt.streamplot(X, Y, U, V, color=U, linewidth=2)

        plt.figure(2)
        plt.title("U and V Interpolated on the center of Pressure control volumes")
        plt.quiver(X, Y, U, V)

        U = np.array(result.ns_x_mesh.phi)
        V = np.array(result.ns_y_mesh.phi)
        U = U.reshape(nx, ny - 1)
        V = V.reshape(nx - 1, ny)

        plt.figure(3)
        plt.title("U velocity in the mesh center-x")
        U_normalized = U / U_bc
        U_center = U_normalized[:, len(U) // 2]
        plt.plot(U_center, np.linspace(0.0, size_y, len(U_center)))

        y_ghia = np.array([
            0.0625,
            0.125 ,
            0.1875,
            0.25  ,
            0.3125,
            0.375 ,
            0.4375,
            0.5   ,
            0.5625,
            0.625 ,
            0.6875,
            0.75  ,
            0.8125,
            0.875 ,
            0.9375,
        ])
        U_ghia = np.loadtxt('ghia_ghia_shin_results/re_10/U.txt')
        plt.plot(U_ghia, y_ghia, 'xb')

        plt.figure(4)
        plt.title("V velocity in the mesh center-y")
        V_normalized = V / U_bc
        V_center = V_normalized[len(V) // 2, :]
        plt.plot(np.linspace(0.0, size_x, len(V_center)), V_center)

        x_ghia = np.array([
            0.0625,
            0.125 ,
            0.1875,
            0.25  ,
            0.3125,
            0.375 ,
            0.4375,
            0.5   ,
            0.5625,
            0.625 ,
            0.6875,
            0.75  ,
            0.8125,
            0.875 ,
            0.9375,
        ])
        V_ghia = np.loadtxt('ghia_ghia_shin_results/re_10/V.txt')
        plt.plot(x_ghia, V_ghia, 'xb')

        plt.show()


if __name__ == '__main__':
    main()
