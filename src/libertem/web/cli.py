import click
import os


@click.command()
@click.option('--port', help='port on which the server should listen on',
              default=9000, type=int)
@click.option('--local-directory', help='local directory to manage dask-worker-space files',
              default='dask-worker-space', type=str)
# FIXME: the host parameter is currently disabled, as it poses a security risk
# as long as there is no authentication
# see also: https://github.com/LiberTEM/LiberTEM/issues/67
# @click.option('--host', help='host on which the server should listen on',
#               default="localhost", type=str)
def main(port, local_directory, host="localhost"):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    from libertem.cli_tweaks import console_tweaks
    from .server import run
    console_tweaks()
    run(host, port, local_directory)
