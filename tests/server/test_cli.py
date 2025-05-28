import asyncio
import signal
import time
import sys

import pytest


@pytest.fixture(scope="module")
def event_loop_policy(request):
    if sys.platform.startswith("win"):
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.get_event_loop_policy()  # TODO: windows fixes


@pytest.mark.asyncio
async def test_libertem_server_cli_startup(http_client):
    CTRL_C = signal.SIGINT
    # make sure we can start `libertem-server` and stop it again using ctrl+c
    # this is kind of a smoke test, which should cover the main cli functions.
    p = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'libertem.web.cli', '--no-browser',
        stderr=asyncio.subprocess.PIPE,
    )
    # total deadline, basically how long it takes to import all the dependencies
    # and start the web API
    # (no executor startup is done here)
    deadline = time.monotonic() + 15
    while True:
        if time.monotonic() > deadline:
            assert False, 'timeout'
        line = await asyncio.wait_for(p.stderr.readline(), 5)
        if not line:  # EOF
            assert False, 'subprocess is dead'
        line = line.decode("utf8")
        print('Line:', line, end='')
        if 'LiberTEM listening on' in line:
            break

    async def _debug():
        while True:
            line = await asyncio.wait_for(p.stderr.readline(), 5)
            if not line:  # EOF
                return
            line = line.decode("utf8")
            print('Line@_debug:', line, end='')

    debug_task = asyncio.ensure_future(_debug())

    try:
        addr = line[line.find("http"):].strip()
        async with http_client.get(addr, timeout=30.0) as response:
            assert response.status == 200, "Failed to GET from libertem-server"

        # windows likes to be special, so we just kill the subprocess instead:
        if sys.platform.startswith("win"):
            if p.returncode is None:
                p.terminate()
                await asyncio.sleep(0.2)
                await asyncio.wait_for(p.wait(), 1)
            if p.returncode is None:
                p.kill()
                await asyncio.wait_for(p.wait(), 1)
        else:
            # now, let's kill the subprocess:
            # ctrl+c twice should do the job:
            p.send_signal(CTRL_C)
            await asyncio.sleep(0.5)
            if p.returncode is None:
                p.send_signal(CTRL_C)

            # wait for the process to stop, but max. 30 second:
            ret = await asyncio.wait_for(p.wait(), 30)

            # in unix, minus signal number is returned if the process was killed:
            assert ret in (0, -CTRL_C)
    except Exception:
        if p.returncode is None:
            p.terminate()
            await asyncio.sleep(0.2)
        if p.returncode is None:
            p.kill()
        await asyncio.wait_for(p.wait(), 1)
        raise
    finally:
        debug_task.cancel()
