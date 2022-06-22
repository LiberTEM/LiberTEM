from typing import Tuple

import click
import logging

from libertem.executor.cli import preload_help

log_values = "Allowed values are 'critical', 'error', 'warning', 'info', 'debug'."


def get_token(token_path):
    token = None
    if token_path is not None:
        with open(token_path) as f:
            token = f.read().strip()
        if len(token) == 0:
            raise click.UsageError(
                f'{token_path} is empty! Refusing to start with insecure configuration.'
            )
    return token


@click.command()
@click.option('-h', '--host', help='host on which the server should listen on',
              default="localhost", type=str, show_default=True)
@click.option('-p', '--port', help='port on which the server should listen on, [default: 9000]',
              default=None, type=int)
@click.option('-d', '--local-directory', help='local directory to manage dask-worker-space files',
              default='dask-worker-space', type=str)
@click.option('-b/-n', '--browser/--no-browser',
              help='enable/disable opening the browser', default='True')
@click.option('-l', '--log-level',
              help=f"set logging level. Default is 'info'. {log_values}",
              default='INFO')
@click.option('-t', '--token-path',
              help="path to a file containing a token for authenticating API requests",
              type=click.Path(exists=True))
@click.option('--preload', help=preload_help, default=(), type=str, multiple=True)
@click.option('--insecure',
              help="allow to bind to non-localhost without token auth",
              default=False, is_flag=True)
def main(port, local_directory, browser, log_level, insecure, host="localhost",
        token_path=None, preload: Tuple[str, ...] = ()):
    is_custom_port = port is not None
    if port is None:
        port = 9000

    from libertem.common.threading import set_num_threads_env
    from libertem.common.tracing import maybe_setup_tracing
    token = get_token(token_path)
    is_localhost = host in ['localhost', '127.0.0.1', '::1']
    if token is None and not is_localhost and not insecure:
        raise click.UsageError(
            f'listening on non-localhost {host}:{port} currently requires token authentication '
            f'or --insecure'
        )
    maybe_setup_tracing(service_name="libertem-server")
    with set_num_threads_env(1):
        from libertem.cli_tweaks import console_tweaks
        from .server import run
        console_tweaks()
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise click.UsageError(f'Invalid log level: {log_level}.\n{log_values}')
        run(host, port, browser, local_directory, numeric_level, token, preload, is_custom_port)
