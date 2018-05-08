import processing
import numpy as np

ds = processing.BinaryHDFSDataSet(index_path="test/index.json", host='localhost', port=8020)
maskcount = 2
masks = np.ones((ds.framesize, maskcount))

job = processing.ApplyMasksJob(dataset=ds, masks=masks)

executor = processing.DaskJobExecutor(scheduler_uri="tcp://localhost:8786")
executor.client.scheduler_info()

sum([len(result) for result in executor.run_job(job)])
