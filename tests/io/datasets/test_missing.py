import inspect
import pytest


from libertem.io.dataset import filetypes, get_dataset_cls


@pytest.mark.parametrize(
    "ds_key", tuple(filetypes.keys())
)
def test_missing(lt_ctx, ds_key):
    ds_class = get_dataset_cls(ds_key)
    classname = ds_class.__name__
    params = inspect.signature(ds_class.__init__).parameters
    for extension in ds_class.get_supported_extensions():
        filename = f"not_a_file.{extension}"
        with pytest.raises(Exception) as e:
            lt_ctx.load(
                ds_key, filename,
            )
        e_msg = str(e.value)

        print(f'{classname} + {filename} => {e.typename}("{e_msg}")')

        if "nav_shape" in e_msg and "nav_shape" in params:
            nav_shape = (8, 8)
            with pytest.raises(Exception) as e:
                lt_ctx.load(
                    ds_key, filename, nav_shape=nav_shape
                )
            e_msg = str(e.value)
            print(f'{classname} + {filename} + nav_shape={nav_shape} => {e.typename}("{e_msg}")')
