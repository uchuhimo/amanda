import reprlib
from typing import Any, Dict, List

import immutables


class Attributes(Dict[str, Any]):
    def __init__(self, col=None, **kw):
        if isinstance(col, Attributes):
            _map = col._map
            if kw:
                _map = _map.update(**kw)
        else:
            if col:
                _map = immutables.Map(col, **kw)
            else:
                _map = immutables.Map(**kw)
        self._map: immutables.Map[str, Any] = _map

    def clear(self):
        self._map = immutables.Map()

    def copy(self) -> "Attributes":
        return Attributes(self)

    def get(self, key, default=None):
        return self._map.get(key, default)

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def pop(self, key, default=None):
        try:
            self._map = self._map.delete(key)
        except KeyError:
            if default is not None:
                return default
            else:
                raise KeyError

    def popitem(self):
        raise NotImplementedError()

    def setdefault(self, key, value):
        if key in self:
            return self[key]
        else:
            self._map = self._map.set(key, value)
            return value

    def update(self, col=None, **kw):
        self._map = self._map.update(col, **kw)

    def values(self):
        return self._map.values()

    def __contains__(self, key):
        return self._map.__contains__(key)

    def __delitem__(self, key):
        self._map = self._map.delete(key)

    def __eq__(self, other):
        if not isinstance(other, Attributes):
            return False
        else:
            return self._map.__eq__(other._map)

    def __getitem__(self, key):
        return self._map.__getitem__(key)

    def __iter__(self):
        return self._map.__iter__()

    def __len__(self):
        return self._map.__len__()

    def to_item_strings(self) -> List[str]:
        items = []
        for name, value in self.items():
            if not name.startswith("/"):
                value_string = repr(value)
                if "\n" in value_string:
                    value_string = value_string.replace("\n", " ")
                items.append(f"{name}={value_string}")
        for name, value in self.items():
            if name.startswith("/"):
                value_string = repr(value)
                if "\n" in value_string:
                    value_string = value_string.replace("\n", " ")
                items.append(f"{name}={value_string}")
        return items

    @reprlib.recursive_repr("{...}")
    def __repr__(self):
        return "Attributes({{{}}})".format(", ".join(self.to_item_strings()))

    def __setitem__(self, key, value):
        self._map = self._map.set(key, value)
