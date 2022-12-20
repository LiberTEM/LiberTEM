# LiberTEM Config TOML Description

## Objectives

- Allow common LT objects (datasets, contexts, analyses, ROIs) to be
  defined using a human-writable configuration file.
- Motivating example: datasets requiring lists of file paths as part
  of their parameters, which don't fit into the standard LT
  DataSet constructor interface. A config sidecar file for the 
  dataset would allow quicker instantiation of these objects, and 
  allow loading such datasets in the web interface.
- Extension: a complete configuration file could allow job submission
  to a persistent LiberTEM processing server

## Requirements

- A configuration for a 'standard' object, one not requiring special parameters, should be expressible with minimal lines:

```toml
format = 'raw'
path = './data.bin'
nav_shape = [100, 100]
sig_shape = [256, 256]
dtype = 'float32'
```

The above snippet should function as a complete descriptor for a 
`RawDataSet` if passed to `ctx.load(toml_path)`.

- Interpretation of the top-level contents of the file can either be:
  - **implicit**: only `ctx.load(toml_path)` will correctly interpret the top-level content in the above snippet as a dataset
  - **explicit**: using a `type` key to let a generic loader read any specification, e.g.:

```toml
type = 'dataset'
format = 'mib'
path = './data.hdr'
```

- Inline definitions are better than requiring a particular structure:

```toml
format = 'glob_dataset'
files = './image*.bin'
```

should be equivalent to:

```toml
format = 'glob_dataset'
files = '#/my_fileset'

[my_fileset]
type='fileset'
files = './image*.bin'
```

if the `glob_dataset` constructor knows that `files` should be interpreted as a `fileset`. This means we don't need to manually
specify a `fileset` object if we don't need to use additional options like controlling file sort order.

- The object being constructed should have a mechanism to both validate and interpret the config file as it wants to, doing so via a JSON-schema specification, e.g.:

```json
{
    "type": "dataset",
    "title": "GLOB dataset",
    "properties": {
        "format": {
            "const": "glob_dataset",
        }
        "files": {
            "type": "fileset",
        },
    },
    "required": ["files"],
}
```

if the `files` property is present and points to a `fileset` object
already, then the schema will validate (with optional validation of a subschema). If the `files` object does not point to a `fileset` object, then:

1. If `files` is itself dictionary-like but not a `fileset` then it will be cast to one and validated according to any subschema.
2. If `files` is not dictionary-like (a string, for example) then
   we will try to construct a `fileset` with this value, if `fileset` supports such a construction. We then apply any subschema to validate the `fileset`.

- Defintions can be nested or can be referenced relative to the root
  of the config file from anywhere in the file:

```toml
[my_dataset]
type = 'dataset'
format = 'glob_dataset'
files = '#/my_fileset'

[my_fileset]
type='fileset'
files = './image*.bin'
```

is interpreted equivalently to:

```toml
[my_dataset]
type = 'dataset'
format = 'glob_dataset'

[my_dataset.files]
type='fileset'
files = './image*.bin'
```

we use the JSON-schema path format `'#/path/to/key'` to specify external objects, always from the root of the file. This is necessary to reduce scope for ambiguity between a string value that could be interpreted as a true value, or a key in the tree.

- The schema can specify simple defaults using the standard schema default key, and the parser will fill these defaults when missing from the config file.

```json
{
  "type": "file",
  "properties": {
      "dtype": {
          "type": "dtype",
          "default": "float32"
      },
  },
  "required": ["dtype"],
}
```

- The consumer of the config is responsible for extracting the information it needs from the tree, once validated/interpreted under the provided schema.

## Implementation

TOML (like other formats such as YAML/JSON) is parsed to a top-level Python 
dictionary with potentially nested child dictionaries.

- Keys must be strings
- TOML can encode values similarly to JSON, with the notable exception of a missing `Null` value (which would deserialize to `None` in Python).

In practice, as all of the interpretation of the file content is after the 
conversion to Python objects, the actual format of the file is not important
and could be TOML, YAML, JSON, pkl etc as long as the top level is a dictionary.

To allow parsing, we treat the nested dictionaries as a tree by subclassing 
`dict` to define `NestedDict`, which gives a child dictionary a reference to 
its parent. This allows us to create sub-configs which can point to elsewhere 
in the tree.

The parsing process:

1. Read the config into a Python dictionary with `tomli`, for example.
2. Traverse the tree to find any `dict` instances, if:
    - The `dict` contains the key `'type'` matching a config we understand, convert it to a `NestedDict`-subclass implementing that `type`
    - Otherwise, convert the `dict` instance to a bare `NestedDict` instance.
3. To ready a config for consumption by `LiberTEM`, we can validate it 
against a schema. This schema is used to validate compatibility, including 
setting default values. It also is used to coerce the types of any values
which did not specify their own type (i.e. bare values or bare `NestedDict`).
4. Depending on the object type, the config can then be consumed by a constructor (e.g. `DataSet.initialize()`), or the config can self-construct
into the object (e.g. `ArraySpec.resolve() -> np.ndarray`).

### Reserved keys

- `type`: identifies a structure as a type we should be able to interpret
- `read_as`: a `type` identifier used to read the content of a structure as if
it were another type, while retaining the identity of the original type.
- `root`: a string defining an absolute directory path from which other, 
relative paths should be resolved


## Features

### File paths
- File paths can be absolute or relative to a root path.
- Root paths are hierarchical and the most closest root specified in a parent takes precendence.
- If no root path is specified, then:
  - the root path is taken to be the directory containing the TOML file (if reading from file) 
  - the current Python working directory (if parsing TOML from a Python string)

Some examples:

```toml
root = '/path/to/dataset/'
format = 'glob_dataset'
# The glob is resolved relative to parent root /path/to/dataset/
files = './image*.bin'

[my_absolute_file]
type = 'file'
path = '/other/path/to/file.raw'

[my_relative_file_parent]
type = 'file'
# Resovles relative to /path/to/dataset/
path = './file.raw'

[my_relative_file_self]
root = '/other/path/to/'
type = 'file'
# Resovles relative to /other/path/to/
path = './file.raw'
```

### Globs and sorting for groups of files

Both single files and groups of files can be specified with glob syntax using the Python `glob` library.

If multiple files match a single-`file` specification, then an error will be raised.

`glob` expansion does not guarantee anything about the order of the returned `fileset`, so to cover this case `fileset` objects support sorting with `natsorted`.

```toml
[my_fileset]
type='fileset'
files = './image*.bin'
sort = 'natsorted' # or 'os_sorted' or 'humansorted'
# optional sort_options in the natsorted.ns enum
sort_options = [
    "NUMAFTER",
    "COMPATIBILITYNORMALIZE"
]
```

The sort options supported are any of the strings in the `natsorted.ns` configuration enum, https://natsort.readthedocs.io/en/5.2.0/ns_class.html#natsort.ns

### Loading of data from files

File-like specifications support an optional `format` key  which can allow them to be loaded directly into a Python object. The most obvious application is to load `np.array` data, which can be specified as such:

```toml
[my_array]
type = 'file'
path = './image.npy'
```

Which when resolved will check against a set of supported formats:

```python
format_defs = {
    'raw': load_raw,
    'bin': load_raw,
    'npy': np.load,
    'tiff': load_image,
    'tif': load_image,
    'jpg': load_image,
    'png': load_image,
}
```

using the file extension (or be read directly from a `format` key if present). Any `load_options` are passed through to the loader, for example:

```toml
[my_raw_array]
type = 'file'
format = 'raw'
path = './data.raw'

[my_raw_array.load_options]
dtype = 'float32'
shape = [64, 64]
```

which will be passed through to `load_raw` to read the raw file with the correct dtype and shape.

By adding additional loaders, this could also support loading non-numpy data.

### Inline specification of array data

Using the `array` type, we can specify array-like values inline:

```toml
[my_list]
type = 'array'
data = [
    [5, 6, 7, 8],
    [1, 2, 3, 4],
]
dtype = 'float32'
```

### Delegation of interpretation

A spec can delegate how it should be interpreted with the `read_as` key. In this case we want to specify an `array` (to match a schema, for example), but we delegate how to load the data by using `read_as` as a `file` spec:

```toml
[my_image]
type = 'array'
read_as = 'file'
path = './image.png'
```

This is useful if we want to define objects which should only be used in specific contexts, e.g.:

```toml
[my_roi]
type = 'roi'
read_as = 'file'
path = './my_roi.npy'
```

which lets us state that the `my_roi` object should be available anywhere an `roi` is needed, but it should be loaded like a file object.

A further example:

```toml
[my_roi]
type = 'roi'
read_as = 'array'
data = [
    [True, False, True],
    [False, True, False],
    [True, False, True],
]
dtype = 'bool'
```

which allows inline specification of an ROI, interpreted like an `array`.

### Direct construction of ROI objects

If all we need is to toggle a few pixels in an ROI, this can be done by specifying these directly:

```toml
[my_roi]
type = 'roi'
shape = [100, 200]
roi_base = True
toggle_px = [
    [10, 56],
    [78, 12],
]
```
which would set the coordinates in `toggle_px` to False (relative to a `roi` which is otherwise True).

It would also be possible to (later) allow the shape parameter to be filled using the dataset shape itself, or load coordinates to toggle from an `array` (e.g. from a text file).


## Usage

This section is the least-well-defined, and consists about ideas for potential usages of these config files.

The most obvious usage is in the definition of DataSets:

#### As the `path` argument for a dataset

Both of the following are equivalent

```python
ctx.load('mib', path='./mib_ds_config.toml')
```
and implicitly:
```python
MIBDataSet('./mib_ds_config.toml')
```

In these cases, the dataset is responsible for loading and 
interpreting the contents of the config file, and handling any 
conflicts between values defined inside it and any other 
potential arguments to its constructor, for example:

```python
ctx.load('mib', path='./mib_ds_config.toml', nav_shape=(50, 50))
```
where `mib_ds_config.toml` contains `nav_shape == (100, 100)`.

To be consistent with past usage, the validation and checking
would likely occur inside `ds.initialize()` rather than in the `__init__`.

The basic usage would be a top-level config like the following:

```toml
path = '/path/to/file.hdr'
nav_shape = [50, 50]
sig_shape = [256, 256]
sync_offset = 0
```

with specification of

- `type = 'dataset'`
- `format = 'mib'`

at the top-level being completely optional, as long as `ctx.load` is 
given the dataset type to try to load.

The following should also be supported:

```toml
[my_dataset]
type = 'dataset'
format = 'mib'
path = '/path/to/file.hdr'
nav_shape = [50, 50]
sig_shape = [256, 256]
sync_offset = 0

[my_other_object]
...
```

when passed to `ctx.load('mib', ...)`, iff one `format='mib'` dataset type config is present in the file.

An ambiguity arises when there is a top-level definition which is not 
compatible with the dataset, but a child structure which is. In this 
case the dataset should first try to use the top-level data, and only 
search the children when this fails.


#### As a parameter for `ctx.load('auto')`

There are two ways to implement this:

1. Using the normal 'auto' logic of trying every dataset in order of 
definition until one matches, but with the config file path rather 
than a dataset file path.
2. Have `ctx.load` read the config file, and require a `format` key 
at the top level or a single `type = 'dataset'` definition with 
`format` key, to know where to dispatch the load command.


#### As a parameter to a generic loader

Fully explicit configs, i.e. those which include a `type` key 
could be used to load objects without needing to use specific 
methods like `ctx.load` for datasets.

A generic loader could directly instantiate one-or-more elements
from the config file. If only one element is found this could be 
returned directly. If multiple elements are found they could be
returned in a dict-like structure matching the config. This structure
could have convenience methods allowing access to views of the data
split by type, e.g.:

```python
loaded = generic_loader('config.toml')
assert isinstance(loaded, dict)

>>> loaded.datasets
{
  'my_mib': MIBDataSet(...),
  'my_raw': RawFileDataSet(...),
}
>>> loaded.contexts
...
>>> loaded.analyses
...
```

#### Support for pre-loaded configs / configs from string

All of the above examples imply a real config file on disk, but it 
would be useful to support in-memory configs, either as a string
form of a serialization language, e.g.
```python
config_string = ('{"type: "dataset",'
                  '"format": "mib",'
                  '"path": "./file.hdr"}')
dataset = generic_loader(config_string, config_format='json')
```
or from a Python object, e.g.:
```python
config = {
  'type': 'dataset',
  'format': 'mib',
  'path': './file.hdr'
}
dataset = generic_loader(config)
```

This would be necessary to support a future job-processing server model.
