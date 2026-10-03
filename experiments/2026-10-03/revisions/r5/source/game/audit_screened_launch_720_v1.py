"""Direct launch of a cold-screened Triton ABI; no runtime imports at module load.

Matches CompiledKernel.__getitem__ in the supplied Intel Triton runtime. Shape,
stride, dtype, device, constexpr values and compiler alignment attributes are
checked before each dispatch. Binary identity stays under the existing child
resource seal. This never compiles or performs a JIT cache query in a frame.
"""


def tensor_abi(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


class ScreenedLaunch:
    def __init__(self, kernel, jit, arguments, grid, tensor_type):
        self.kernel, self.jit, self.tensor_type = kernel, jit, tensor_type
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        if len(self.grid) != 3:
            raise ValueError('Only a three-dimensional compiled grid is supported')
        self.abi = tuple(('tensor', tensor_abi(a)) if isinstance(a, tensor_type) else
                         ('constant', type(a), a) for a in arguments)
        src = kernel.src
        if src.fn is not jit or len(src.fn.arg_names) != len(arguments):
            raise RuntimeError('Cold screened compiled source/argument arity changed')
        self.constants = tuple(src.constants.items())
        self.alignments = []
        for path, attributes in src.attrs.items():
            if len(path) != 1:
                raise RuntimeError('Nested compiled argument ABI is not reviewed')
            for name, value in attributes:
                if name != 'tt.divisibility' or type(value) is not int or value < 1:
                    raise RuntimeError('Unreviewed compiler argument attribute')
                self.alignments.append((path[0], value))
        self.alignments = tuple(self.alignments)
        # Create the runner while cold. All original launch instrumentation
        # hooks and the runtime's native launch/stream resolution stay active.
        self.runner = kernel[self.grid]

    def check(self, jit, arguments, grid):
        grid = tuple(grid) + (1,) * (3 - len(grid))
        if self.jit is not jit or grid != self.grid or len(arguments) != len(self.abi):
            raise RuntimeError('Screened launch JIT/grid/argument arity changed')
        for argument, row in zip(arguments, self.abi):
            if row[0] == 'tensor':
                if not isinstance(argument, self.tensor_type) or tensor_abi(argument) != row[1]:
                    raise RuntimeError('Screened launch tensor shape/stride/dtype/device changed')
            elif type(argument) is not row[1] or argument != row[2]:
                raise RuntimeError('Screened launch constant changed')
        for path, value in self.constants:
            if len(path) != 1 or arguments[path[0]] != value:
                raise RuntimeError('Screened constexpr specialization changed')
        for index, alignment in self.alignments:
            argument = arguments[index]
            pointer = argument.data_ptr() if isinstance(argument, self.tensor_type) else argument
            if pointer % alignment:
                raise RuntimeError('Screened launch compiler alignment requirement changed')

    def __call__(self, jit, arguments, grid):
        self.check(jit, arguments, grid)
        self.runner(*arguments)
        return self.kernel
