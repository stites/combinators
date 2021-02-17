""" type aliases """

from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from combinators.stochastic import Trace, Factor, GenericRandomVariable
import combinators.tensor.utils as tensor_utils
import inspect


class property_dict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        return "<property_dict>:\n" + show_nest(self, pd_header="<property_dict>")

def show_nest(p:property_dict, nest_level=0, indent_len:Optional[int]=None, pd_header="<property_dict>"):
    _max_len = max(map(len, p.keys()))
    max_len = _max_len + nest_level * (_max_len if indent_len is None else indent_len)
    delimiter = "\n  "
    def showitem(v):
        if isinstance(v, Tensor):
            return tensor_utils.show(v)
        elif isinstance(v, dict):
            return "dict({})".format(", ".join(["{}={}".format(k, showitem(v)) for k, v in v.items()]))
        else:
            return repr(v)

    unnested = dict(filter(lambda kv: not isinstance(kv[1], property_dict), p.items()))
    unnested_str = delimiter.join([
        *[("{:>"+ str(max_len)+ "}: {}").format(k, showitem(v)) for k, v in unnested.items()
         ]
    ])

    nested = dict(filter(lambda kv: isinstance(kv[1], property_dict), p.items()))
    nested_str = delimiter.join([
        *[("{:>"+ str(max_len)+ "}: {}").format(k + pd_header, "\n"+show_nest(v, nest_level=nest_level+1)) for k, v in nested.items()
         ]
    ])

    return unnested_str + delimiter + nested_str

PropertyDict = property_dict

class iproperty_dict(property_dict):
    def __iter__(self):
        for v in self.values():
            yield v

IPropertyDict = iproperty_dict

class Out(PropertyDict):
    def __init__(self, trace:Trace, log_weight:Optional[Tensor], output:Any, extras:dict=dict()):
        self.trace = trace       # τ; ρ
        self.log_weight = log_weight # w      FIXME: this should be log weight
        self.output = output     # c
        for k, v in extras.items():
            self[k] = v

    def __iter__(self):
        optionals = ['ix', 'program', 'kernel', 'proposal']
        extras = dict()
        for o in optionals:
            if hasattr(self, o):
                extras[o] = getattr(self, o)

        for x in [self.trace, self.log_weight, self.output, extras]:
            yield x