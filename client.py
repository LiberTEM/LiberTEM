import os
import subprocess
import pathlib
import shutil
import click
import logging


logger = logging.getLogger(pathlib.Path(__file__).name)
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    pass


@cli.command(name="build")
def build_client():
    # build the js client
    cwd = os.path.dirname(__file__)
    cwd_client = os.path.join(cwd, 'client')
    logger.info(
        "building js client",
    )
    npm = shutil.which('npm')
    for command in [[npm, 'install'],
                    [npm, 'run-script', 'build']]:
        logger.info(' '.join(command))
        subprocess.check_call(command, cwd=cwd_client)
    _copy_client()


def _copy_client():
    # copy the js client
    cwd = pathlib.Path(__file__).absolute().parent
    cwd_client = cwd / 'client'
    client = cwd / 'src' / 'libertem' / 'web' / 'client'

    logger.info(
        "preparing output directory: %s" % client,
    )
    if client.exists():
        shutil.rmtree(client)

    build = cwd_client / "dist"
    logger.info(
        f"copying client: {build} -> {client}",
    )
    shutil.copytree(build, client)


@cli.command(name="copy")
def copy_client():
    _copy_client()


if __name__ == '__main__':
    cli()
