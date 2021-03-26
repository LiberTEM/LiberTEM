from multiprocessing import shared_memory as shm
import json
from io import BytesIO
import time
import pickle
import hashlib
import warnings

import sparse
import numpy as np

from libertem.common.container import MaskContainer


# Since shared memory may be returned in page-sized blocks
# instead of the original size and the remainder might be uninitialized,
# we encode the size in the first 8 bytes to make sure we decode and load only
# "good" data
def meta_to_bytes(obj):
    enc = json.dumps(obj).encode('utf-8')
    # prepend length encoded as as int64
    sizebytes = bytes(np.int64([len(enc)])) + enc
    return sizebytes


def meta_from_bytes(b):
    offset = np.dtype(np.int64).itemsize
    size = np.ndarray(shape=1, dtype=np.int64, buffer=b)[0]
    return json.loads(str(b[offset:offset+size], encoding='utf-8'))


# We (ab)use the npz format to encode the array including metadata in memory
def numpy_to_bytes(a):
    b = BytesIO()
    np.lib.format.write_array(b, a)
    return b.getbuffer()


def numpy_from_bytes(b):
    b_io = BytesIO(b)
    magic = np.lib.format.read_magic(b_io)
    if magic == (1, 0):
        header = np.lib.format.read_array_header_1_0(b_io)
    elif magic == (2, 0):
        header = np.lib.format.read_array_header_2_0(b_io)
    else:
        raise ValueError(f"Encountered unknown magic {magic}")
    offset = b_io.tell()
    return np.ndarray(shape=header[0], order=('F' if header[1] else 'C'), dtype=header[2], buffer=b, offset=offset)


def store(computed_masks, base_key, ref_queue):
    if isinstance(computed_masks, sparse.COO):
        kind = 'coo'
    elif isinstance(computed_masks, np.ndarray):
        kind = 'ndarray'
    else:
        raise TypeError(f"Don't know how to handle {type(computed_masks)}")
    meta = {
        'kind': kind
    }
    meta_bytes = meta_to_bytes(meta)
    meta_shm = shm.SharedMemory(create=True, name=f"{base_key}_meta", size=len(meta_bytes))
    ref_queue.put(meta_shm.name)
    meta_shm.buf[:] = meta_bytes

    if kind == 'coo':
        coords_bytes = numpy_to_bytes(computed_masks.coords)
        coords_shm = shm.SharedMemory(create=True, name=f"{base_key}_coords", size=len(coords_bytes))
        ref_queue.put(coords_shm.name)
        coords_shm.buf[:] = coords_bytes

        data_bytes = numpy_to_bytes(computed_masks.data)
        data_shm = shm.SharedMemory(create=True, name=f"{base_key}_data", size=len(data_bytes))
        ref_queue.put(data_shm.name)
        data_shm.buf[:] = data_bytes
        result = (meta_shm, coords_shm, data_shm)
    else:
        data_bytes = numpy_to_bytes(computed_masks)
        data_shm = shm.SharedMemory(create=True, name=f"{base_key}_data", size=len(data_bytes))
        ref_queue.put(data_shm.name)
        data_shm.buf[:] = data_bytes
        result = (meta_shm, data_shm)
    return result


def load(base_key):
    meta_shm = shm.SharedMemory(create=False, name=f"{base_key}_meta")
    meta = meta_from_bytes(meta_shm.buf)
    kind = meta['kind']

    if kind == 'coo':
        coords_shm = shm.SharedMemory(create=False, name=f"{base_key}_coords")
        coords = numpy_from_bytes(coords_shm.buf)

        data_shm = shm.SharedMemory(create=False, name=f"{base_key}_data")
        data = numpy_from_bytes(data_shm.buf)
        result = (sparse.COO(data=data, coords=coords), [meta_shm, coords_shm, data_shm])
    elif kind == 'ndarray':
        data_shm = shm.SharedMemory(create=False, name=f"{base_key}_data")
        data = numpy_from_bytes(data_shm.buf)
        result = (data, [meta_shm, data_shm])
    else:
        raise ValueError(f"Unknown kind {kind}")
    return result


def load_or_create(mask_factories, ref_queue, salt=0):
    id = hashlib.sha512(pickle.dumps(mask_factories)).hexdigest()
    key = f"{salt}_{id}"
    try:
        m = shm.SharedMemory(create=True, name=f"{key}", size=1)
        ref_queue.put(m.name)
        c = MaskContainer(mask_factories=mask_factories)
        # "handle" has to stay in scope until the data is loaded
        # so that the reference count doesn't go down to 0
        _ = store(c.computed_masks, key, ref_queue)
        ref_queue.join()
        print(f"stored {key}")
        obj, handles = load(key)
        handles.append(m)
        # Make sure all references are kept save, i.e. the "keepalive"
        # process has finished creating a reference
        # Keep track of all existing handles
        return (obj, handles)
    except FileExistsError as e:
        print(e)
        for i in range(10):
            try:
                print(f"loading {key}...")
                m = shm.SharedMemory(create=False, name=f"{key}")
                obj, handles = load(key)
                handles.append(m)
                return (obj, handles)
            except FileNotFoundError as e:
                print(e)
                time.sleep(0.1)
        # Fallback: Calculate locally
        c = MaskContainer(mask_factories=mask_factories)
        print(f"fallback {key}")
        return (c.computed_masks, [])


def keep_a_reference(q):
    refs = []
    while True:
        name = q.get()
        try:
            print(f"Trying to keep a reference of {name}...")
            ref = shm.SharedMemory(create=False, name=name)
            print(f"Reference of {name}: {ref}")
            refs.append(ref)
        except FileNotFoundError as e:
            warnings.warn(str(e))
        q.task_done()
