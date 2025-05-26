import logging
import socket

import click

log_values = "Allowed values are 'critical', 'error', 'warning', 'info', 'debug'."


preload_help = (
    'Module, file or code to preload on workers, for example HDF5 plugins. '
    'Can be specified multiple times. See also '
    'https://docs.dask.org/en/stable/customize-initialization.html#preload-scripts '
    'for the behavior with Dask workers (current default)'
    'and https://libertem.github.io/LiberTEM/reference/dataset.html#hdf5 '
    'for information on loading HDF5 files that depend on custom filters.'
)


@click.command()
@click.argument('scheduler', default="tcp://localhost:8786", type=str)
@click.option('-k', '--kind', help='Worker kind. Currently only "dask" is implemented.',
              default="dask", type=str)
@click.option('-d', '--local-directory', help='local directory to manage temporary files',
              default='dask-worker-space', type=str)
@click.option('-c', '--n-cpus', type=int, default=None,
              help='Number of CPUs to use, defaults to number of CPU cores without hyperthreading.')
@click.option('-u', '--cudas', type=str, default=None,
              help='List of CUDA device IDs to use, defaults to all detected CUDA devices. '
              'Use "" to deactivate CUDA.')
@click.option('-p', '--has-cupy', type=bool, default=None,
              help='Activate CuPy integration, defaults to detection of installed CuPy module.')
@click.option('-n', '--name', help='Name of the cluster node, defaults to host name',
              type=str)
@click.option('-l', '--log-level', help=f"set logging level. Default is 'info'. {log_values}",
              default='INFO')
@click.option('--preload', help=preload_help,
              default=None, type=str, multiple=True)
def main(kind, scheduler, local_directory, n_cpus, cudas,
         has_cupy, name, log_level, preload: tuple[str, ...]):
    from libertem.common.threading import set_num_threads_env
    with set_num_threads_env(1):
        from libertem.utils.devices import detect
        from libertem.cli_tweaks import console_tweaks
        from libertem.executor.dask import cli_worker
        console_tweaks()

        if kind != 'dask':
            raise NotImplementedError(
                f"Currently only worker kind 'dask' is implemented, got {kind}."
            )

        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise click.UsageError(f'Invalid log level: {log_level}.\n{log_values}')

        defaults = detect()

        if n_cpus is None:
            cpus = list(defaults['cpus'])
        else:
            cpus = list(range(n_cpus))

        if cudas == '':
            cudas = []
        elif cudas is None:
            cudas = list(defaults['cudas'])
        else:
            cudas = list(map(int, cudas.split(',')))

        if has_cupy is None:
            has_cupy = defaults['has_cupy']

        if not name:
            name = socket.gethostname()

        cli_worker(
            scheduler=scheduler,
            local_directory=local_directory,
            cpus=cpus,
            cudas=cudas,
            has_cupy=has_cupy,
            name=name,
            log_level=numeric_level,
            preload=preload,
        )
