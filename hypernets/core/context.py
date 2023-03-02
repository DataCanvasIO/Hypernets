import abc


class Context(metaclass=abc.ABCMeta):

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError


class DefaultContext(Context):

    def __init__(self):
        super(DefaultContext, self).__init__()
        self._map = {}

    def put(self, key, value):
        self._map[key] = value

    def get(self, key):
        return self._map.get(key)

    # def __getstate__(self):
    #     states = dict(self.__dict__)
    #     if '_map' in states:  # mark _map as transient
    #         states['_map'] = {}
    #     return states
