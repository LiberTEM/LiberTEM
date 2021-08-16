import click
import logging

log_values = "Allowed values are 'critical', 'error', 'warning', 'info', 'debug'."


@click.command()
@click.option('-p', '--port', help='port on which the server should listen on',
              default=9000, type=int)
@click.option('-d', '--local-directory', help='local directory to manage dask-worker-space files',
              default='dask-worker-space', type=str)
@click.option('-b/-n', '--browser/--no-browser',
              help='enable/disable opening the browser', default='True')
@click.option('-l', '--log-level', help=f"set logging level. Default is 'info'. {log_values}",
              default='INFO')
# FIXME: the host parameter is currently disabled, as it poses a security risk
# as long as there is no authentication
# see also: https://github.com/LiberTEM/LiberTEM/issues/67
# @click.option('--host', help='host on which the server should listen on',
#               default="localhost", type=str)
def main(port, local_directory, browser, log_level, host="localhost"):
    from libertem.utils.threading import set_num_threads_env
    with set_num_threads_env(1):
        from libertem.cli_tweaks import console_tweaks
        from .server import run
        console_tweaks()
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise click.UsageError(f'Invalid log level: {log_level}.\n{log_values}')
        run(host, port, browser, local_directory, numeric_level)
