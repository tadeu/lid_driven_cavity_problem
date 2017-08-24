from experiments.run_solver import run_solver


class TimeSuite:
    def setup(self):
        pass

    def time_scipy_python_10(self):
        run_solver(solver_type='scipy', kernel_type='python', nx=10, ny=10)

    def time_petsc_python_30(self):
        run_solver(solver_type='petsc', kernel_type='python', nx=30, ny=30)

    def time_petsc_numpy_30(self):
        run_solver(solver_type='petsc', kernel_type='numpy', nx=30, ny=30)

    def time_petsc_cython_30(self):
        run_solver(solver_type='petsc', kernel_type='cython', nx=30, ny=30)
