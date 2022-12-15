import pathlib
from typing import Dict, Any, Optional

from utils import MissingKey, ParserException, find_tree_root, resolve_jsonpath
from validation import get_validator


class NestedDict(dict):
    """
    Nested dictionary class with knowledge of its parent in the tree
    Implements two features:

    - the ability to resolve keys within the tree using JSON syntax
      relative to the root
            #/path/to/key
      This could be extended to resolve using posix path semantics
    - The ability to search upwards in the tree for a specific key
    """
    def _set_parent(self, parent: Dict[str, Any]):
        self._parent = parent

    @property
    def parent(self):
        try:
            return self._parent
        except AttributeError:
            return None

    def resolve_key(self, key: str):
        """
        Get key from tree in JSON path notation
           i.e. #/key1/key2
        starting from root
        If not available then raise
        """
        if not isinstance(key, str) or not key.startswith('#'):
            raise TypeError(f'Cannot resolve key {key}')
        root = find_tree_root(self)
        return resolve_jsonpath(root, key)

    def resolve_upwards(self, key):
        value = self.get(key, MissingKey())
        if isinstance(value, MissingKey):
            if self.parent is not None:
                return self.parent.resolve_upwards(key)
            else:
                raise ParserException(f'Cannot resolve {key} in tree')
        return value


class SpecBase(NestedDict):
    spec_type = 'base'

    def __init__(self, **kvals):
        super().__init__(self, **kvals)
        if not self.validate(None, self):
            raise ParserException(f'Invalid spec for {self}')

    @property
    def root(self) -> Optional[pathlib.Path]:
        root = self.resolve_upwards('root')
        try:
            return pathlib.Path(root)
        except TypeError:
            return root

    def load(self):
        # Try to load the oject defined by this spec
        # Will call load on all sub-specs (assumed to be required)
        raise NotImplementedError(f'No load method for {self.__class__.__name__}')

    @classmethod
    def construct(cls, arg, parent=None):
        if isinstance(arg, dict):
            instance = cls(**arg)
            # Retain arg's parent while casting if it
            # is already a NestedDict, even if None,
            # else use the external parent
            try:
                instance._set_parent(arg.parent)
            except AttributeError:
                instance._set_parent(parent)
            return instance
        else:
            raise ParserException(f'Unrecognized spec {arg} for {cls.__name__}')

    @classmethod
    def validate(cls, checker, instance):
        if instance.read_as is not None and instance.read_as not in instance.readers():
            return False
        return isinstance(instance, cls)

    @property
    def read_as(self):
        return self.get('read_as', None)

    def view(self, spec_type: str, read_as: str = None):
        """
        Get a copy of this instance as another type

        Removes the read_as key from the copy
        and sets 'type' to equal the new type
        """
        from parser import spec_types
        if spec_type not in spec_types:
            raise ParserException(f'Cannot view {self.__class__.__name__} as {spec_type}')
        instance_props = {k: v for k, v in self.items() if k != 'read_as'}
        if 'type' in instance_props:
            instance_props['type'] = spec_type
        if read_as is not None:
            instance_props['read_as'] = read_as
        instance = spec_types[spec_type](**instance_props)
        instance._set_parent(self.parent)
        return instance

    def check_schema(self, schema: Dict[str, Any]):
        from parser import spec_types, extra_types
        validator = get_validator(schema, {**spec_types, **extra_types})
        validator.validate(self)

    @classmethod
    def readers(cls):
        return {}

    def resolve(self):
        raise NotImplementedError('Cannot resolve bare SpecBase')

    def resolve_as(self, spec_type):
        if spec_type in self.readers():
            return self.readers()[self.read_as](self)
        raise ParserException(f'Unrecognized read_as "{self.read_as}" for '
                              f'{self.__class__.__name__}')
