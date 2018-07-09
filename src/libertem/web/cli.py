import click
from .server import run


@click.command()
@click.option('--port', help='port on which the server should listen on',
              default=9000, type=int)
@click.option('--host', help='host on which the server should listen on',
              default="localhost", type=str)
def main(host, port):
    run(host, port)
