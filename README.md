#  vector

This package contains three simple tensor-based vector implementations:
- `vector.tensor`: a vector of simple 1D tensors that are stored contiguously
- `vector.atomic`: a vector of tensors of arbitrary sizes
- `vector.string`: a vector of strings that does not use Lua memory

All `vector` objects are fully serializable and can be stored in `tds` data
structures.

Simple example use cases are provided below. For more detailed examples of
the usage of these `vectors`, please refer to the unit test.

## vector.tensor

The most common use case of `vector.tensor` is for storing an unknown number of
numbers or tensors of variable sizes, and collapsing the result into a single
tensor:

```
vector = require 'vector'
t = vector.tensor.new_double()
t[1] = math.random()
t[#t + 1] = 2
t:insertTensor(torch.randn(2, 2))
t[#t + 1] = -5
print(#t)
for k, v in pairs(t) do print(k, v) end
print(t:getTensor())
```

All these operations are efficient and perform a minimal amount of reallocation.
The following tensor types are supported: `char`, `byte`, `short`, `int`,
`long`, `float`, and `double`.

## vector.atomic

The most common use case of `vector.atomic` is for storing an unknown number of
tensors of variable sizes:

```
vector = require 'vector'
t = vector.atomic.new_double()
t[1] = torch.randn(2)
t[2] = torch.randn(3, 2)
t[3] = torch.randn(1, 1, 2)
print(#t)
for k, v in pairs(t) do print(k, v) end
```

In contrast to `fb.atomic` containers, this vector implementation is readily
serializable and can be stored in `tds` data structures. The underlying data
structure stores all tensors in a single storage. This allows for much faster
(de)serialization than if the tensors were stored in a `tds` object.

The following tensor types are supported: `char`, `byte`, `short`, `int`,
`long`, `float`, and `double`.

## vector.string

The most common use case of `vector.string` is for storing an unknown number of
strings without consuming Lua memory:

```
vector = require 'vector'
t = vector.string.new()
t[1] = 'lorem'
t[2] = 'ipsum'
t[3] = 'dolor'
print(#t)
for k, v in pairs(t) do print(k, v) end
```

The underlying data structure stores all tensors in a single `CharStorage`. This
allows for much faster (de)serialization than if the tensors were stored in a
`tds` object or Lua table. The memory used to store the strings does not count
against Lua memory limits (in case a 32-bit version of LuaJIT is used).
