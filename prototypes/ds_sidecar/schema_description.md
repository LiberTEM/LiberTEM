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

- A configuration for a 'standard' object, one not requiring special
parameters, should be expressible with minimal lines:

```toml
ds_format = 'raw'
path = './data.bin'
nav_shape = [100, 100]
sig_shape = [256, 256]
dtype = 'float32'
```

The above snippet should function as a complete descriptor for a 
`RawDataSet` if passed to `ctx.load(toml_path)`.

- Interpretation of the top-level contents of the file can either be:
  - **implicit**: only `ctx.load(toml_path)` will correctly interpret
  the top-level content in the above snippet as a dataset
  - **explicit**: using a `config_type` key to let a generic loader read
  any specification, e.g.:

```toml
config_type = 'dataset'
ds_format = 'mib'
path = './data.hdr'
```

- Inline definitions are better than requiring a particular structure:

```toml
ds_format = 'glob_dataset'
files = './image*.bin'
```

should be equivalent to:

```toml
ds_format = 'glob_dataset'
files = '#/my_fileset'

[my_fileset]
config_type='fileset'
files = './image*.bin'
```

if the `glob_dataset` constructor knows that `files` should be
interpreted as a `fileset`. This means we don't need to manually
specify a `fileset` object if we don't need to use additional
options like controlling file sort order.

- The object being constructed should have a mechanism to both
validate and interpret the config file as it wants to, doing
so using a Pydantic Model, e.g.:

```python
class GlobDatasetModel(BaseModel):
    ds_format: Optional[Literal['glob_dataset']] = 'glob_dataset'
    files: Union[
              FilesetConfig,
              List[pathlib.Path],
              List[str],
              pathlib.Path,
              str,
          ]
```

if the `files` property is present and points to a `FilesetConfig` object
already, then the schema will validate. Else:

1. If `files` is itself dictionary-like but not a `FilesetConfig` then it
will be validated against the `FilesetConfig` model.
2. If `files` is not dictionary-like (a string, for example) then
   we will try to construct a `FilesetConfig` with this value, if `fileset`
   supports such a construction.

- Defintions can be nested or can be referenced relative to the root
  of the config file from anywhere in the file:

```toml
[my_dataset]
config_type = 'dataset'
ds_format = 'glob_dataset'
files = '#/my_fileset'

[my_fileset]
config_type='fileset'
files = './image*.bin'
```

is interpreted equivalently to:

```toml
[my_dataset]
config_type = 'dataset'
ds_format = 'glob_dataset'

[my_dataset.files]
config_type='fileset'
files = './image*.bin'
```

we use the JSON-schema path format `'#/path/to/key'` to specify external
paths, always from the root of the file. This is necessary to reduce
scope for ambiguity between a string value that could be interpreted
as a value itself, or a key in the tree.

- The consumer of the config is responsible for extracting the information
it needs from the tree, once validated/interpreted under the provided schema.
- As the schema is a Pydantic Model and therefore a Python class, it can have both dot attributes `arrayconfig.dtype` read directly from the input data as well as methods, such as `file.path.resolve()` to resolve a path.

## Implementation

TOML (like other formats such as YAML/JSON) is parsed to a top-level Python 
dictionary with potentially nested child dictionaries.

- Keys must be strings
- TOML can encode values similarly to JSON, with the notable exception of a
missing `Null` value (which would deserialize to `None` in Python).

In practice, as all of the interpretation of the file content is after the 
conversion to Python objects, the actual format of the file is not important
and could be TOML, YAML, JSON, pkl etc as long as the top level is a dictionary.

To allow parsing, we treat the nested dictionaries as a tree by subclassing 
`dict` to define `NestedDict`, which gives a child dictionary a reference to 
its parent. This allows us to create sub-configs which can point to elsewhere 
in the tree.

The parsing process:

1. Read the config into a Python dictionary with `tomli`, for example.
2. Traverse the tree to find any `dict` instances and convert them to `NestedDict` instances.
3. Consumers of the config supply the schema / model they want to validate against. They can apply this schema to all sub-trees as a search to get all valid configs, or the user can supply the unique (sub-)tree to check against the schema.

### Reserved keys

- `config_type`: identifies a structure as a type we should be able to interpret
- `root`: a string defining an absolute directory path from which other, 
relative paths should be resolved. If not specified at the top of the config file, this is automatically set to the location of the file. All sub-trees inherit the root of their parent (allowing roots to be hierarchical).


## Features

### File paths
- File paths can be absolute or relative to a root path.
- Root paths are hierarchical and the most closest root specified
in a parent takes precendence.
- If no root path is specified, then:
  - the root path is taken to be the directory containing the TOML
  file (if reading from file) 
  - the current Python working directory (if parsing TOML from a Python string)

Some examples:

```toml
root = '/path/to/dataset/'
ds_format = 'glob_dataset'
# The glob is resolved relative to parent root /path/to/dataset/
files = './image*.bin'

[my_absolute_file]
config_type = 'file'
path = '/other/path/to/file.raw'

[my_relative_file_parent]
config_type = 'file'
# Resovles relative to /path/to/dataset/
path = './file.raw'

[my_relative_file_self]
root = '/other/path/to/'
config_type = 'file'
# Resovles relative to /other/path/to/
path = './file.raw'
```

### Globs and sorting for groups of files

Both single files and groups of files can be specified with
glob syntax using the Python `glob` library.

If multiple files match a single-`file` specification, then an
error will be raised.

`glob` expansion does not guarantee anything about the order of
the returned `fileset`, so to cover this case `fileset` objects
support sorting with `natsorted`, which is the default.

```toml
[my_fileset]
config_type='fileset'
files = './image*.bin'
sort = 'natsorted' # or 'os_sorted' or 'humansorted' or 'none'
# optional sort_options in the natsorted.ns enum
sort_options = [
    "NUMAFTER",
    "COMPATIBILITYNORMALIZE"
]
```

The sort options supported are any of the strings in the 
`natsorted.ns` configuration enum, https://natsort.readthedocs.io/en/5.2.0/ns_class.html#natsort.ns

### Loading of data from files

File-like specifications support an optional `format` key  which
can allow them to be loaded directly into a Python object. The most
obvious application is to load `np.array` data, which can be specified as such:

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

using the file extension (or be read directly from a `format`
key if present). Any `load_options` are passed through to the
loader, for example:

```toml
[my_raw_array]
config_type = 'file'
format = 'raw'
path = './data.raw'

[my_raw_array.load_options]
dtype = 'float32'
shape = [64, 64]
```

which will be passed through to `load_raw` to read the raw file
with the correct dtype and shape.

By adding additional loaders, this could also support loading non-numpy data.

### Inline specification of array data

Using the `array` type, we can specify array-like values inline:

```toml
[my_list]
config_type = 'array'
data = [
    [5, 6, 7, 8],
    [1, 2, 3, 4],
]
dtype = 'float32'
shape = [1, 8]
```


## Usage

This section is the least-well-defined, and consists about
ideas for potential usages of these config files.

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
config_type = 'dataset'
ds_format = 'mib'
path = '/path/to/file.hdr'
nav_shape = [50, 50]
sig_shape = [256, 256]
sync_offset = 0

[my_other_object]
...
```

when passed to `ctx.load('mib', ...)`, iff one `format='mib'` dataset
type config is present in the file.

An ambiguity arises when there is a top-level definition which is not 
compatible with the dataset, but a child structure which is. In this 
case the dataset should first try to use the top-level data, and only 
search the children when this fails.


#### As a parameter for `ctx.load('auto')`

There are two ways to implement this:

1. Using the normal 'auto' logic of trying every dataset in order of 
definition until one matches, but with the config file path rather 
than a dataset file path.
2. Have `ctx.load` read the config file, and require a `ds_format` key 
at the top level or a single `config_type = 'dataset'` definition with 
`ds_format` key, to know where to dispatch the load command.


#### As a parameter to a generic loader

Fully explicit configs, i.e. those which include `config_type` keys
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
config_string = ('{"config_type: "dataset",'
                  '"ds_format": "mib",'
                  '"path": "./file.hdr"}')
dataset = generic_loader(config_string, config_format='json')
```
or from a Python object, e.g.:
```python
config = {
  'config_type': 'dataset',
  'ds_format': 'mib',
  'path': './file.hdr'
}
dataset = generic_loader(config)
```

This would be necessary to support a future job-processing server model.

To simplify the implementation it is useful to only support this form when using
the generic loader else we have to pass extra arguments to constructors
like `ctx.load` in order to correctly interpret the string syntax.