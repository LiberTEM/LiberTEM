import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from functools import partial
import asyncio
import time
import sys
import multiprocessing

import numpy as np
import psutil
import dask.distributed as dd

from libertem import api
from libertem.io.dataset import load
from libertem.executor.dask import AsyncDaskJobExecutor, DaskJobExecutor
from libertem.job.sum import SumFramesJob


async def background_task():
    # this loop exits when Task.cancel() is called
    while True:
        print("DoEvents")
        await asyncio.sleep(1)


def get_result_buffer(job):
    empty = np.zeros(job.get_result_shape())
    return empty


async def run(executor, job, out):
    # print("run entered")
    async for tiles in executor.run_job(job):
        # print("Tiles")
        for tile in tiles:
            tile.copy_to_result(out)
        yield out
    # print("Run finished")


async def async_main(address):
    # start background task: (can be replaced with asyncio.create_task(coro) in Python 3.7)
    background_events = asyncio.ensure_future(background_task())
    
    executor = await AsyncDaskJobExecutor.connect(address)

    ds = load(
        "blo",
        path=("C:/Users/weber/Nextcloud/Projects/Open Pixelated STEM framework/"
        "Data/3rd-Party Datasets/Glasgow/10 um 110.blo"),
        tileshape=(1,8,144,144)
    )

    job = SumFramesJob(dataset=ds)

    out = get_result_buffer(job)

    async for part_result in run(executor, job, out):
        print("Partial result sum: ", out.sum())

    print("Final result sum: ", out.sum())

    # stop the background task:
    background_events.cancel()


def main():
    cores = psutil.cpu_count(logical=False)

    if cores is None:
        cores = 2
    cluster_kwargs = {
        "threads_per_worker": 1,
        # "asynchronous": True,
        "n_workers": cores
    }

    cluster = dd.LocalCluster(**cluster_kwargs)
    loop = asyncio.get_event_loop()

    try:
        # print(cluster.scheduler_address)
        # (can be replaced with asyncio.run(coro) in Python 3.7)
        loop.run_until_complete(async_main(cluster.scheduler_address))
        
    finally:
        # loop.close()
        print("Close cluster")
        cluster.close()


if __name__ == "__main__":
    main()
