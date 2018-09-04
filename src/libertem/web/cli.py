import click
from libertem.cli_tweaks import console_tweaks
from .server import run


@click.command()
@click.option('--port', help='port on which the server should listen on',
              default=9000, type=int)
# FIXME: the host parameter is currently disabled, as it poses a security risk
# as long as there is no authentication
# see also: https://github.com/LiberTEM/LiberTEM/issues/67
# @click.option('--host', help='host on which the server should listen on',
#               default="localhost", type=str)
def main(port, host="localhost"):
    console_tweaks()
    run(host, port)
