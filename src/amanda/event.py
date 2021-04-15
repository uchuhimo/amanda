from dataclasses import dataclass
from typing import Callable, Iterable
from enum import Enum, auto
import logging

@dataclass(frozen=True)
class Event:
    name: str


on_graph_loaded = Event("on_graph_loaded")
before_graph_executed = Event("before_graph_executed")
after_graph_executed = Event("after_graph_executed")
after_backward_graph_executed = Event("after_backward_graph_executed")
before_subgraph_executed = Event("before_subgraph_executed")
after_subgraph_executed = Event("after_subgraph_executed")
after_backward_subgraph_executed = Event("after_backward_subgraph_executed")
before_op_executed = Event("before_op_executed")
after_op_executed = Event("after_op_executed")
before_backward_op_executed = Event("before_backward_op_executed")
after_backward_op_executed = Event("after_backward_op_executed")

EventCallback = Callable[["EventContext"], None]

class UpdateStatus(Enum):
    No_Grad = auto()
    Not_Updated = auto()
    Is_Updated = auto()

class GradStatue():
    def __init__(self, event: Event, status: UpdateStatus) -> None:
        self.event = event
        self.status = status


class EventContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool]):
        super(EventContext, self).__init__()
        from amanda.tool import Tool

        self.tools: Iterable[Tool] = tools
        self.bw_hooks = []

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.get_callback(event)(self)

    def bw_callback(self, event, key_pair, grad):
        context_key, arg_key = key_pair
        if not arg_key==None:
            self[context_key][arg_key] = grad
        else:
            self[context_key] = grad
        if self.all_grad_updated(event):
            self.trigger(event)
            # self.remove_bw_hooks()


    def registry_backward_callback(self, event: Event, **kwargs) -> None:
        def register_bw_hook(context, event, key_pair, grad):
            context.bw_callback(event, key_pair, grad)

        for key,value in kwargs.items():
            if key == 'args':
                self['grad_args'] = [None for i in range(len(value))]
                for args_index, args_value in enumerate(value):
                    if hasattr(args_value, 'register_hook') and args_value.requires_grad:
                        hook = args_value.register_hook(lambda grad,args_index=args_index: register_bw_hook(self, event, ('grad_args', args_index), grad))
                        self.bw_hooks.append(hook)
                        self['grad_args'][args_index] = event
                    else:
                        self['grad_args'][args_index] = None
                continue

            if key == 'kwargs':
                self['grad_kwargs'] = dict()
                for args_index, args_value in value.items():
                    if hasattr(args_value, 'register_hook') and args_value.requires_grad:
                        hook = args_value.register_hook(lambda grad,args_index=args_index: register_bw_hook(self, event, ('grad_kwargs', args_index), grad))
                        self.bw_hooks.append(hook)
                        self['grad_kwargs'][args_index] = event
                    else:
                        self['grad_kwargs'][args_index] = None
                continue

            if hasattr(value, 'register_hook') and value.requires_grad:
                hook = value.register_hook(lambda grad: register_bw_hook(self, event, ('grad_'+key, None), grad))
                self.bw_hooks.append(hook)
                self.update({'grad_'+key:event})
            else:
                self.update({'grad_'+key:None})

        
    def all_grad_updated(self, event: Event) -> bool:
        for key, value in self.items():
            if key == 'grad_args':
                for args_value in value:
                    if isinstance(args_value, Event) and args_value == event:
                        return False
                continue
            if key == 'grad_kwargs':
                for args_value in value.values():
                    if isinstance(args_value, Event) and args_value == event:
                        return False
                continue
            if key.startswith('grad_') and isinstance(value, Event) and value == event:
                return False
        return True



    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False

    def remove_bw_hooks(self):
        for handle in self.bw_hooks:
            handle.remove()