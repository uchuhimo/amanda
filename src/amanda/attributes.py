import reprlib
from typing import Mapping

import immutables


class Attributes(Mapping):
    def __init__(self, col=None, **kw):
        self._map: immutables.Map
        if isinstance(col, Attributes):
            self._map = col._map
            if kw:
                self._map = self._map.update(**kw)
        else:
            if col:
                self._map = immutables.Map(col, **kw)
            else:
                self._map = immutables.Map(**kw)

    def clear(self):
        self._map = immutables.Map()

    def copy(self):
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

    @reprlib.recursive_repr("{...}")
    def __repr__(self):
        items = []
        for key, val in self.items():
            items.append("{!r}: {!r}".format(key, val))
        return "<Attributes({{{}}})>".format(", ".join(items))

    def __setitem__(self, key, value):
        self._map = self._map.set(key, value)
