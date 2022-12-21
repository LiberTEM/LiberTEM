import pathlib
from typing import Dict, Any, Optional, TYPE_CHECKING, Callable, Generator

from utils import MissingKey, ParserException, resolve_jsonpath
from validation import get_validator


if TYPE_CHECKING:
    from jsonschema.validators import Draft202012Validator


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

    @property
    def root(self) -> 'NestedDict':
        parent = self.parent
        if parent is not None:
            return parent.root
        return self

    @property
    def where(self) -> str:
        """
        Get the JSON path #/ from root for this struct
        """
        parent = self.parent
        if parent is not None:
            me, = tuple(k for k, v in parent.items() if v is self)
            me = f'{parent.where}/{me}'
        else:
            me = '#/'
        return me

    def resolve_key(self, key: str):
        """
        Get key from tree in JSON path notation
           i.e. #/key1/key2
        starting from root
        If not available then raise
        """
        if not isinstance(key, str):
            raise TypeError(f'Invalid key {key}')
        if not key.startswith('#/'):
            raise ValueError(f'Can only resolve keys in JSON-path syntax (#/), got {key}')
        return resolve_jsonpath(self.root, key)

    def resolve_upwards(self, key):
        value = self.get(key, MissingKey())
        if isinstance(value, MissingKey):
            if self.parent is not None:
                return self.parent.resolve_upwards(key)
            else:
                raise ParserException(f'Cannot resolve {key} in tree')
        return value

    def copy(self):
        """
        Return a copy of this NestedDict instance where
        all the .parent references also point to copies in a new tree
        """
        root = self.root
        new_root = root._copy_down()
        return new_root.resolve_key(self.where)

    def _copy_down(self):
        new = self.__class__(**self)
        copy = {}
        for key, value in self.items():
            try:
                value = value._copy_down()
                value._set_parent(new)
            except AttributeError:
                pass
            copy[key] = value
        new.update(copy)
        return new

    def search(self,
               predicate: Callable[['NestedDict'], bool]) -> Generator['NestedDict', None, None]:
        """
        Depth-first search yielding items matching predicate
        """
        if predicate(self):
            yield self
        for value in self.values():
            if isinstance(value, NestedDict):
                yield from value.search(predicate)


class SpecBase(NestedDict):
    spec_type = 'base'
    resolve_to = None

    def __init__(self, **kvals):
        super().__init__(self, **kvals)
        if not self.validate(None, self):
            raise ParserException(f'Invalid spec for {self}')

    @classmethod
    def validate(cls, checker: Optional['Draft202012Validator'], instance: 'SpecBase') -> bool:
        """
        Called both by SpecBase.__init__ and used
        by an instance of a custom jsonschema.validators.Validator
        to verify that the instance conforms to a schema

        It is expected that child Spec definitions call
        this method with super() to confirm compliance with
        any read_as key and that the instance is indeed of this class
        """
        if instance.read_as is not None and instance.read_as not in instance.readers():
            return False
        return isinstance(instance, cls)

    def apply_schema(self, schema: Dict[str, Any]):
        """
        Method to apply a schema to a SpecBase instance using
        a custom jsonschema Validator

        In applying the schema, keys will be resolved, types
        will be coerced / inferred and defaults will be set
        """
        from spec_tree import spec_types, extra_types
        all_types = {**spec_types, **extra_types}
        new_instance = self.copy()
        # The validator will check and coerce property types but not the root type
        validator = get_validator(schema, all_types)
        validator.validate(new_instance)
        return new_instance

    @classmethod
    def construct(cls, arg, parent=None):
        """
        Construct an instance of this SpecBase from an argument,
        and set its parent as if it were in the tree

        This implementation is the fallback where arg must be a dictionary
        so that its components can be inserted directly into the Spec

        Child SpecBase definitions should define their own constructors
        for differents possible types of arg

        This is used when applying a schema to a non-dict value
        that should be interpreted as a SpecBase, e.g.:

            files = '/*.jpg'

        combined with the schema

            {
                properties: {
                    'files': {'type': 'fileset'}
                }
            }

        will call FileSetSpec.construct('/*.jpg')
        """
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

    @property
    def read_as(self) -> Optional[str]:
        """
        Getter for the 'read_as' key, if present

        If set, must match one key in cls.readers() in order
        to pass cls.validate(..., instance)

        Used to get a view of this instance as another SpecBase
        so that we can use its methods to load data
        """
        return self.get('read_as', None)

    @classmethod
    def readers(cls) -> Dict[str, Callable]:
        """
        Dictionary of accepted read_as keys mapping to
        methods used to interpret the current SpecBase
        as another type

        This method should ideally be a @classproperty but
        that is hard to implement in Python
        """
        return {}

    def view(self, spec_type: str, read_as: str = None) -> 'SpecBase':
        """
        Get a copy of this instance as another type

        Will removes the read_as key from the copy
        and sets 'type' to equal the new type

        This does not modify the tree
        """
        from spec_tree import spec_types
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

    def resolve(self):
        """
        Convert from SpecBase to an instance of the 'resolve_to'
        property of the class

        e.g. convert from a SpecBase defining a file path
        to an instance of pathlib.Path that points at that file
        """
        raise NotImplementedError('Cannot resolve bare SpecBase')

    @property
    def path_root(self) -> Optional[pathlib.Path]:
        """
        Find the root path key upwards in the tree
        in order to resolve relative paths

        This key is either set in the config or set
        by SpecTree to the config file parent directory
        As a fallback the current Python working directory
        is used.
        """
        try:
            root = self.resolve_upwards('root')
        except ParserException:
            # Fallback, though this should have been set by SpecTree
            root = '.'
        try:
            return pathlib.Path(root)
        except TypeError:
            return root
