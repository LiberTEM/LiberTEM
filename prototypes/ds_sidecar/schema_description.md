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

using the file extension (or be read directly from a `format` key if present). Any further parameters can be passed through to the loader, for example:

```toml
[my_raw_array]
type = 'file'
format = 'raw'
path = './data.raw'
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
