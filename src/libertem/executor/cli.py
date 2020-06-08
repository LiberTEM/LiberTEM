import click
import logging

from libertem.utils.devices import detect

log_values = "Allowed values are 'critical', 'error', 'warning', 'info', 'debug'."


@click.command()
@click.argument('scheduler', default="http://localhost:8786", type=str)
@click.option('-k', '--kind', help='Worker kind. Currently only "dask" is implemented.',
              default="dask", type=str)
@click.option('-d', '--local-directory', help='local directory to manage temporary files',
              default='dask-worker-space', type=str)
@click.option('-c', '--n-cpus', type=int, default=0,
              help='Number of CPUs to use, defaults to number of CPU cores without hyperthreading.')
@click.option('-u', '--cudas', type=str, default=None,
              help='List of CUDA device IDs to use, defaults to all detected CUDA devices. '
              'Use "" to deactivate CUDA.')
@click.option('-l', '--log-level', help=f"set logging level. Default is 'info'. {log_values}",
              default='INFO')
def main(kind, scheduler, local_directory, n_cpus, cudas, log_level):
    from libertem.cli_tweaks import console_tweaks
    if kind != 'dask':
        raise NotImplementedError(f"Currently only worker kind 'dask' is implemented, got {kind}.")
    from libertem.executor.dask import cli_worker
    console_tweaks()
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise click.UsageError(f'Invalid log level: {log_level}.\n{log_values}')
    defaults = detect()
    if n_cpus:
        cpus = list(range(n_cpus))
    else:
        cpus = list(defaults['cpus'])

    if cudas == '':
        cudas = []
    elif cudas is None:
        cudas = list(defaults['cudas'])
    else:
        cudas = list(map(int, cudas.split(',')))

    cli_worker(scheduler, local_directory, cpus, cudas, numeric_level)
