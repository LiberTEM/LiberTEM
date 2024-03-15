from typing import Optional
import os
import platform

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
@click.option('-c', '--cpus', help='number of cpu worker processes to use,[default: select in GUI]',
              default=None, type=int)
@click.option('-g', '--gpus', help='number of gpu worker processes to use,[default: select in GUI]',
              default=None, type=int)
@click.option('-o', '--open-ds', help='dataset to preload via URL action',
              default=None, type=str)
@click.option('-l', '--log-level',
              help=f"set logging level. Default is 'info'. {log_values}",
              default='INFO')
@click.option('-t', '--token-path',
              help="path to a file containing a token for authenticating API requests",
              type=click.Path(exists=True))
@click.option('--preload', help=preload_help, default=(), type=str, multiple=True)
@click.option('--insecure',
              help=(
                  "Allow connections from non-localhost without token authorization. "
                  "Applies only when combined with --host <address> "
                  "(use `--insecure -h 0.0.0.0` to accept any connection)"
              ),
              default=False, is_flag=True)
@click.option('--snooze-timeout', type=float,
              help=(
                'Free resources after periods of no activity, in minutes. '
                'Depending on your system, re-starting these resources might '
                'take some time, so typical values are 10 to 30 minutes.'
              ),
              default=None)
def main(port, local_directory, browser, cpus, gpus, open_ds, log_level,
         insecure, host="localhost", token_path=None, preload: tuple[str, ...] = (),
         snooze_timeout: Optional[float] = None):
    # Mitigation for https://stackoverflow.com/questions/71283820/
    #   directory-parameter-on-windows-has-trailing-backslash-replaced-with-double-quote
    if (open_ds and platform.system() == 'Windows' and open_ds[-1] == '"'
            and not os.path.exists(open_ds) and os.path.exists(open_ds[:-1])):
        open_ds = open_ds[:-1]
    # Mitigation for https://github.com/python/cpython/issues/88141
    if platform.system() == 'Windows':
        # Replace the mimetype for .js, as there can be potentially broken
        # entries in the windows registry:
        import mimetypes
        mimetypes.add_type('application/javascript', '.js', strict=True)
    is_custom_port = port is not None
    if port is None:
        port = 9000

    executor_spec = None
    if cpus is not None or gpus is not None:
        executor_spec = dict(
            cpus=cpus if cpus is not None else 0,
            cudas=gpus if gpus is not None else 0,
        )

    from libertem.common.threading import set_num_threads_env
    from libertem.common.tracing import maybe_setup_tracing
    token = get_token(token_path)
    from .server import is_localhost
    if token is None and not is_localhost(host) and not insecure:
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
        if snooze_timeout is not None:
            snooze_timeout *= 60.0
        run(
            host, port, browser, local_directory, numeric_level,
            token, preload, is_custom_port, executor_spec, open_ds,
            snooze_timeout,
        )


# to enable calling this without using the entry point script,
# for example using `python -m libertem.web.cli`:
if __name__ == "__main__":
    main()
