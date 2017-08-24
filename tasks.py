from invoke import Collection, task
from subprocess import call
import sys

@task
def cython(ctx):
    ctx.run('python setup.py build_ext --inplace')

@task
def benchmarks(ctx, quick=False):
    ctx.run('asv run --show-stderr --python=same' + (' --quick' if quick else ''))


ns=Collection()
ns.add_task(cython)
ns.add_task(benchmarks)

# Workaround for Windows execution
if sys.platform.startswith('win'):
    from os import environ
    ns.configure({'run': {'shell': environ['COMSPEC']}})
