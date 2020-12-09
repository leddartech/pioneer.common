from typing import TypeVar, Generic

T = TypeVar('T')

class Option(Generic[T]):
    
    __create_key = object()

    def __init__(self, create_key, item: T = None):
        assert(create_key == Option[T].__create_key), \
            "Result objects must be created using Option.some or Option.none"
        
        self.item = item

    @classmethod
    def some(cls, item: T):
        return Option[T](cls.__create_key, item)

    @classmethod
    def none(cls):
        return Option[T](cls.__create_key)

    def is_some(self) -> bool:
        return self.item is not None

    def is_none(self) -> bool:
        return self.item is None

    def match(self, some_function, none_function):
        if self.is_some():
            return some_function(self.item)
        else:
            return none_function()

    def map(self, map_function):
        if self.is_some():
            return Option.some(map_function(self.item))
        else:
            return self

    def bind(self, bind_function):
        if self.is_some():
            return bind_function(self.item)
        else:
            return self

    def tee(self, side_effect_function):
        if self.is_some():
            side_effect_function(self.item)
        return self

class OptionUtils():

    @staticmethod
    def map(option:Option[T], map_function):
        if option.is_some():
            return Option.some(map_function(option.item))
        else:
            return option

    @staticmethod
    def bind(option: Option[T], bind_function):
        if option.is_some():
            return bind_function(option.item)
        else:
            return option




