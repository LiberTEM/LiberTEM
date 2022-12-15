from file_spec import SpecTree
from validation import get_validator


toml_def = """
[my_metadata_file]
type = 'file'
path = "/home/alex/Data/TVIPS/rec_20200623_080237_000.tvips"
format = 'RAW'
dtype = 'float32'
shape = [64, 64]

[my_data_fileset]
type='fileset'
files = "./testfiles/file*.raw"
sort = false
sort_options = []

[my_data_fileset2]
type='fileset'
files = [
    "./testfiles/file*.raw",
    "./testfiles/file_other*.raw"
]
sort = false
sort_options = []

[my_tvips_dataset]
type = "dataset"
format = "tvips"
meta = '#/my_metadata_file'
data = '#/my_data_fileset'
nav_shape = [32, 32]
sig_shape = [32, 32]
dtype = 'float32'
sync_offset = 0

[my_dark_frame]
type = 'nparray'
data = [
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
]
shape = [2, 8]

[my_roi]
type = 'roi'
read_as = 'file'
path = './test_roi.npy'

[my_roi2]
type = 'roi'
read_as = 'nparray'
data = [
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
]
shape = [2, 8]

[my_corrections]
type='correctionset'

[my_corrections.dark_frame]
type = 'nparray'
data = [
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
]
shape = [2, 8]

[my_corrections.gain_map]
type = 'file'
path = './test_roi.npy'
"""

tvips_schema = {
    "type": "dataset",
    "title": "TVIPS dataset",
    "properties": {
        "meta": {
            "type": "file",
            "properties": {
                "dtype": {
                    "type": "dtype",
                    "default": "float32"
                },
            },
            "required": ["dtype"],
        },
        "data": {
            "type": "fileset",
        },
        "nav_shape": {
            "$ref": "#/$defs/shape",
        },
        "sig_shape": {
            "$ref": "#/$defs/shape",
        },
        "dtype": {
            "type": "dtype",
            "default": "float32"
        },
        "sync_offset": {
            "type": "integer",
            "default": 0,
        },
    },
    "required": ["meta"],
    "$defs": {
        "shape": {
            "type": "array",
            "items": {
                "type": "number",
                "minimum": 1
            },
            "minItems": 1,
        }
    }
}

if __name__ == '__main__':
    nest = SpecTree.from_string(toml_def)
    validator = get_validator(tvips_schema)
    validator.validate(nest['my_tvips_dataset'])
