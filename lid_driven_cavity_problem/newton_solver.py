from lid_driven_cavity_problem.residual_function import residual_function
from copy import deepcopy
from scipy.optimize.minpack import fsolve
from scipy.optimize.slsqp import approx_jacobian
import numpy as np

PLOT_JACOBIAN = False
SHOW_SOLVER_DETAILS = True
IGNORE_DIVERGED = False

class SolverDivergedException(RuntimeError):
    pass


def _create_X(U, V, P):
    return U + V + P


def _plot_jacobian(graph, X):
    import matplotlib.pyplot as plt
    J = approx_jacobian(X, residual_function, 1e-4, graph)
    J = J.astype(dtype='bool')
    plt.imshow(J)
    plt.show()
    

def solve(graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh
    X = _create_X(U=ns_x_mesh.phi, V=ns_x_mesh.phi, P=pressure_mesh.phi)

    if PLOT_JACOBIAN:
        _plot_jacobian(graph, X)

    X_, infodict, ier, mesg = fsolve(residual_function, X, args=(graph,), full_output=True)
    if SHOW_SOLVER_DETAILS:
        print("Number of function calls=%s" % (infodict['nfev'],))
        if ier == 1:
            print("Converged")
        else:
            print("Diverged")
            print(mesg)

    if not IGNORE_DIVERGED:
        if not ier == 1:
            raise SolverDivergedException()
    
    U = X_[0:len(ns_x_mesh)]
    V = X_[len(ns_x_mesh):len(ns_x_mesh) + len(ns_y_mesh)]
    P = X_[len(ns_x_mesh) + len(ns_y_mesh):len(ns_x_mesh) + len(ns_y_mesh) + len(pressure_mesh)]
 
    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]
 
    return new_graph


def solve_using_petsc(graph):
    from petsc4py import PETSc

    options = PETSc.Options()
    options.clear()

    options.setValue('mat_fd_type', 'ds')
#     options.setValue('mat_fd_type', 'wp')
    options.setValue('snes_test_err', '1e-4')
    options.setValue('mat_fd_coloring_err', '1e-4')
    options.setValue('mat_fd_coloring_umin', '1e-4')
    options.setValue('mat_fd_coloring_view', '::ascii_info')

#     options.setValue('log_view', '')
#     options.setValue('log_all', '')

#     options.setValue('pc_type', 'lu')
#     options.setValue('ksp_type', 'preonly')
#     options.setValue('pc_factor_shift_type', 'NONZERO')

    options.setValue('ksp_type', 'gmres')
    options.setValue('pc_type', 'none')

    options.setValue('ksp_max_it', '300')
    options.setValue('ksp_atol', '1e-5')
    options.setValue('ksp_rtol ', '1e-5')
    options.setValue('ksp_divtol ', '1e-5')

    options.setValue('snes_type', 'newtonls')

#     options.setValue('snes_linesearch_type', 'bt')
    options.setValue('snes_linesearch_type', 'basic')

    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh
    X = _create_X(U=ns_x_mesh.phi, V=ns_x_mesh.phi, P=pressure_mesh.phi)

    if PLOT_JACOBIAN:
        _plot_jacobian(graph, X)

    COMM = PETSc.COMM_WORLD

    N = len(X)  # Degrees of freedom of the whole system

    # Creates the Jacobian matrix structure.
    J = PETSc.Mat().createAIJ(N, comm=COMM)

    J.setPreallocationNNZ(12)

    # Must have values in diagonal because of ILU preconditioner
    for i in range(N):
        J.setValue(i, i, 1.0)

    # Creates the Jacobian 4 times, because of upwind asymmetries
    j_structure = np.zeros((N, N), dtype=bool)
    for u_sign in [-1.0, 1.0]:
        for v_sign in [-1.0, 1.0]:
            fake_U = [u_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_x_mesh.phi))]
            fake_V = [v_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_y_mesh.phi))]
            fake_P = [1e3 * (i + 1) ** 2 for i in range(len(pressure_mesh.phi))]
            fake_X = _create_X(fake_U, fake_V, fake_P)
            current_j_structure = approx_jacobian(fake_X, residual_function, 1e-4, graph).astype(dtype='bool')
            j_structure = np.bitwise_or(j_structure, current_j_structure)

#     import matplotlib.pyplot as plt
#     plt.imshow(j_structure)
#     plt.show()

    for i, j in zip(*np.nonzero(j_structure)):
        J.setValue(i, j, 1.0)

    # TODO: Cache this J, it is being recreated at every timestep!
    
#     for i in range(N):
#         for j in range(N):
#             J.setValue(i, j, 0.0)  # The value here doesn't matter, only the "structure"

    J.setUp()
    J.assemble()

    # Data manager structure needed only for calculating the Jacobian automatically using
    # finite differences with "matrix coloring" optimization
    dm = PETSc.DMShell().create(comm=COMM)
    dm.setMatrix(J)  # the Jacobian structure goes inside it

    snes = PETSc.SNES().create(comm=COMM)

    x = PETSc.Vec().createSeq(N)  # solution vector
    # the initial guess goes in the solution vector too
    x.setArray(X)


    b = PETSc.Vec().createSeq(N)  # right-hand side
    b.set(0)

    def residual_function_for_petsc(snes, x, f):
        '''
        Wrapper over our `residual_function` so that it's in a way expected by PETSc.
        '''
        x = x[:]  # transforms `PETSc.Vec` into `numpy.ndarray`
        print('max(x):', max(x))
        f[:] = residual_function(x, graph)
        f.assemble()

    r = PETSc.Vec().createSeq(N)  # residual vector
    snes.setFunction(residual_function_for_petsc, r)

    snes.setDM(dm)

    # Configure it to calculate the Jacobian automatically using finite differences
    snes.setUseFD(True)

    snes.setConvergenceHistory()
    snes.setFromOptions()
    
#     snes.ksp.pc.setType('none')


    snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=50)

    snes.solve(b, x)
    rh, ih = snes.getConvergenceHistory()

    print('(residual, number of linear iterations)')
    print('\n'.join(str(h) for h in zip(rh, ih)))

    if SHOW_SOLVER_DETAILS:
        print("Number of function calls=%s" % (snes.getFunctionEvaluations()))
        
        REASONS = {
            0: 'still iterating',
            # Converged
            2: '||F|| < atol',
            3: '||F|| < rtol',
            4: 'Newton computed step size small; || delta x || < stol || x ||',
            5: 'maximum iterations reached',
            7: 'trust region delta',
            # Diverged
            -1: 'the new x location passed to the function is not in the domain of F',
            -2: 'maximum function count reached',
            -3: 'the linear solve failed',
            -4: 'norm of F is NaN',
            - 5: 'maximum iterations reached',
            -6: 'the line search failed',
            -7: 'inner solve failed',
            -8: '|| J^T b || is small, implies converged to local minimum of F()',
        }
        
        if snes.reason > 0:
            print("Converged with reason:", REASONS[snes.reason])
        else:
            print("Diverged with reason:", REASONS[snes.reason])
            if snes.reason == -3:
                print("Linear solver divergence reason code:", snes.ksp.reason)

    if not IGNORE_DIVERGED:
        if snes.reason <= 0:
            raise SolverDivergedException()

    U = x[0:len(ns_x_mesh)]
    V = x[len(ns_x_mesh):len(ns_x_mesh) + len(ns_y_mesh)]
    P = x[len(ns_x_mesh) + len(ns_y_mesh):len(ns_x_mesh) + len(ns_y_mesh) + len(pressure_mesh)]

    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]

    return new_graph
