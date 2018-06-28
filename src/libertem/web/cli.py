import click
from .server import run


@click.command()
@click.option('--port', help='port on which the server should listen on', default=9000, type=int)
def main(port):
    run(port)
