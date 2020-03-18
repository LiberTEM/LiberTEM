import click


@click.command()
@click.option('--port', help='port on which the server should listen on',
              default=9000, type=int)
@click.option('--local-directory', help='local directory to manage dask-worker-space files',
              default='dask-worker-space', type=str)
@click.option('--browser/--no-browser', help='enable/disable opening the browser', default='True')
# FIXME: the host parameter is currently disabled, as it poses a security risk
# as long as there is no authentication
# see also: https://github.com/LiberTEM/LiberTEM/issues/67
# @click.option('--host', help='host on which the server should listen on',
#               default="localhost", type=str)
def main(port, local_directory, browser, host="localhost"):
    from libertem.cli_tweaks import console_tweaks
    from .server import run
    console_tweaks()
    run(host, port, browser, local_directory)
