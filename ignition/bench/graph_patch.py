"""Clone diagnostic CUDA graphs and replace source-compiled kernels with an unchanged ABI."""

from collections import deque

import torch
from cuda.bindings import driver as cuda


def call(function, *args):
    status, *values = function(*args)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{function.__name__}: {status.name}")
    return values[0] if len(values) == 1 else tuple(values)


def parameter_layout(function):
    result = []
    for index in range(256):
        status, offset, size = cuda.cuFuncGetParamInfo(function, index)
        if status == cuda.CUresult.CUDA_ERROR_INVALID_VALUE:
            return tuple(result)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuFuncGetParamInfo parameter={index}: {status.name}")
        result.append((offset, size))
    raise ValueError("diagnostic kernel replacement supports fewer than 256 parameters")


def ordered_kernels(graph, names):
    _, count = call(cuda.cuGraphGetNodes, graph)
    nodes, count = call(cuda.cuGraphGetNodes, graph, count)
    _, _, _, count = call(cuda.cuGraphGetEdges, graph)
    sources, destinations, _, count = call(cuda.cuGraphGetEdges, graph, count)
    successors = {int(node): [] for node in nodes}
    incoming = dict.fromkeys(successors, 0)
    targets = {}
    handles = {int(node): node for node in nodes}
    for node in nodes:
        if call(cuda.cuGraphNodeGetType, node) == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = call(cuda.cuGraphKernelNodeGetParams, node)
            if int(params.func):
                name = call(cuda.cuFuncGetName, params.func).decode()
                if name in names:
                    targets[int(node)] = (name, params)
    for source, destination in zip(sources, destinations):
        successors[int(source)].append(int(destination))
        incoming[int(destination)] += 1
    ready = deque(node for node, count in incoming.items() if not count)
    ancestors = dict.fromkeys(incoming, 0)
    ordered, visited = [], 0
    while ready:
        node = ready.popleft()
        visited += 1
        if node in targets:
            if ordered and not ancestors[node] & (1 << (len(ordered) - 1)):
                raise ValueError("replacement kernels must form one dependency-ordered chain")
            name, params = targets[node]
            ordered.append((handles[node], name, params))
            ancestors[node] |= 1 << (len(ordered) - 1)
        for child in successors[node]:
            ancestors[child] |= ancestors[node]
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("captured CUDA graph is not acyclic")
    return ordered


class KernelGraph:
    """A cloned executable borrowing the baseline's persistent and graph-local storage."""

    def __init__(self, source, replacements):
        self.source, self.replacements = source, tuple(replacements)
        self.graph = self.executable = None
        self.device = torch.cuda.current_device()
        original = cuda.CUgraph(source.raw_cuda_graph())
        if not int(original):
            raise ValueError("kernel replacement requires a retained CUDA graph template")
        names = {name for name, _ in self.replacements}
        nodes = ordered_kernels(original, names)
        if [name for _, name, _ in nodes] != [name for name, _ in self.replacements]:
            raise ValueError("captured kernel chain differs from the explicit replacement sequence")
        try:
            self.graph = call(cuda.cuGraphClone, original)
            for (node, name, params), (_, replacement) in zip(nodes, self.replacements, strict=True):
                compiled = replacement.compiled
                compiled._init_handles()
                function = cuda.CUfunction(compiled.function)
                if parameter_layout(params.func) != parameter_layout(function):
                    raise ValueError(f"replacement kernel {name} changes the parameter ABI")
                if (tuple(replacement.grid) != (params.gridDimX, params.gridDimY, params.gridDimZ)
                        or compiled.metadata.num_ctas != 1):
                    raise ValueError(f"replacement kernel {name} changes grid geometry or requires clusters")
                clone = call(cuda.cuGraphNodeFindInClone, node, self.graph)
                params.func = function
                params.blockDimX, params.blockDimY, params.blockDimZ = compiled.metadata.num_warps * 32, 1, 1
                params.sharedMemBytes = compiled.metadata.shared
                call(cuda.cuGraphKernelNodeSetParams, clone, params)
            self.executable = call(cuda.cuGraphInstantiate, self.graph, 0)
        except BaseException:
            self.reset()
            raise

    def replay(self):
        if self.executable is None:
            raise RuntimeError("diagnostic graph has been released")
        call(cuda.cuGraphLaunch, self.executable, cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))

    def reset(self):
        torch.cuda.synchronize(self.device)
        if self.executable is not None:
            call(cuda.cuGraphExecDestroy, self.executable)
            self.executable = None
        if self.graph is not None:
            call(cuda.cuGraphDestroy, self.graph)
            self.graph = None
        self.source = None
        self.replacements = ()
