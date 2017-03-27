--[[

This is a simple atomic vector implementation on top of Torch, which supports
all Tensor types. It is serializable so you can save it in a file, and it can be
stored in tds.hash. The Tensors can have arbitrary dimensionality and you can
insert a mix of Tensors with different dimensionalities and sizes. The main
limitation is that it only supports sequential insertion.

Note: Whilst this implementation does support 1D CharTensors, it does not
automatically convert Lua strings to 1D CharTensors or vice-versa. If you want
to push Lua strings onto a vector, please use `stringvector` instead.

]]--


-- dependencies:
require 'torch'
local ffi  = require 'ffi'
local _tds = require 'tds'
local elem = require 'tds.elem'

-- function that defines AtomicVector for a specific type:
local function defineAtomicTensorVector(typename)

    -- define C type:
    local tensorname  = string.format('%sTensor',         typename)
    local storagename = string.format('%sStorage',        typename)
    local vectorname  = string.format('%sAtomicVector',   typename)
    local cvectorname = string.format('TH%sAtomicVector', typename)
    local storagefree = string.format('TH%sStorage_free', typename)
    local cdef = [[
        typedef struct {
          THRealStorage *content;
          THLongStorage *pointers;
          THLongStorage *sizes;
          THLongStorage *sizeIndex;
          THLongStorage *numElements;
        } THRealAtomicVector;

        void THRealStorage_free(THRealStorage *storage);
        void THLongStorage_free(THLongStorage *storage);

        void *malloc(size_t size);
        void free(void *ptr);
    ]]
    cdef = cdef:gsub('Real', typename)
    ffi.cdef(cdef)

    -- register C-structure in tds:
    local mt = {}
    local function free_p(celem)    -- frees the C-structure
        celem = ffi.cast(cvectorname .. '*', celem)
        ffi.C[storagefree](celem.content)
        ffi.C['THLongStorage_free'](celem.pointers)
        ffi.C['THLongStorage_free'](celem.sizes)
        ffi.C['THLongStorage_free'](celem.sizeIndex)
        ffi.C['THLongStorage_free'](celem.numElements)
        ffi.C.free(celem)
    end
    free_p = ffi.cast('void (*)(void*)', free_p)
    local function set_func(lelem)  -- sets the C-structure
        local celem = ffi.cast(
            cvectorname .. '*',
            ffi.C.malloc(ffi.sizeof(cvectorname))
        )
        local fields = {
            'content', 'numElements', 'pointers', 'sizes', 'sizeIndex'
        }
        for _,val in pairs(fields) do
            celem[val] = lelem[val]:cdata()
            lelem[val]:retain()
        end
        return celem, free_p
    end
    local function get_func(celem)   -- gets the C-structure
        local celem = ffi.cast(cvectorname .. '*', celem)
        local lelem = {}
        local fields = {'numElements', 'pointers', 'sizes', 'sizeIndex'}
        for _,val in pairs(fields) do
            lelem[val] = torch.pushudata(celem[val], 'torch.LongStorage')
            lelem[val]:retain()
        end
        lelem.content  = torch.pushudata(
            celem.content,  'torch.' .. storagename
        )
        setmetatable(lelem, mt)
        return lelem
    end
    elem.addctype(  -- adds type as a tds element
        vectorname,
        free_p,
        set_func,
        get_func
    )

    -- define all functions in the metatable:
    function mt.__new()
        local self = {}
        self.content = torch[storagename](1)
        self.pointers = torch.LongStorage(1)
        self.sizes = torch.LongStorage(1)
        self.sizeIndex = torch.LongStorage(1)
        self.pointers[1] = 1
        self.sizeIndex[1] = 1
        self.numElements = torch.LongStorage(1)
        self.numElements[1] = 0    -- numbers don't work with tds.hash
        setmetatable(self, mt)
        return self
    end

    function mt:__index(k)
        assert(self)
        if type(k) == 'string' then return rawget(mt, k) end
        if k <= 0 or k > self.numElements[1] then
            error('index out of bounds')
        end
        local size = (self.sizeIndex[k + 1] - self.sizeIndex[k] > 0) and
            torch.LongTensor(self.sizes):narrow(
                1, self.sizeIndex[k], self.sizeIndex[k + 1] - self.sizeIndex[k]
            ) or torch.LongTensor()
        return (size:nElement() == 0 or size:prod() == 0)
            and torch[tensorname]()
            or  torch[tensorname](self.content):narrow(
                    1, self.pointers[k], size:prod()
                ):resize(torch.LongStorage(size:totable()))
    end

    function mt:__newindex(k, v)
        assert(self)
        if k ~= self.numElements[1] + 1 then
            error('only sequential writing')
        end
        if not v then error('removal not supported') end
        if torch.typename(v) ~= 'torch.' .. tensorname then
            error(string.format('can only insert %s elements', tensorname))
        end

        -- compute sizes of current content:
        local contentSize = (self.numElements[1] == 0) and 0
            or self.pointers[self.numElements[1] + 1] - 1
        local sizeIndexSize = (self.numElements[1] == 0) and 0
            or self.sizeIndex[self.numElements[1] + 1] - 1

        -- resize buffers if necessary:
        while contentSize + v:nElement() > self.content:size() do
            self.content:resize(self.content:size() * 2)
        end
        while sizeIndexSize + v:nDimension() > self.sizes:size() do
            self.sizes:resize(self.sizes:size() * 2)
        end
        if k > (self.pointers:size() - 1) then
            self.pointers:resize(self.pointers:size() * 2)
        end
        if k > (self.sizeIndex:size() - 1) then
            self.sizeIndex:resize(self.sizeIndex:size() * 2)
        end

        -- insert new tensor:
        self.pointers[ k + 1] = self.pointers[k]  + v:nElement()
        self.sizeIndex[k + 1] = self.sizeIndex[k] + v:nDimension()
        if v:nElement() > 0 then
            torch[tensorname](self.content):narrow(
                1, self.pointers[k], self.pointers[k + 1] - self.pointers[k]
            ):copy(v)
            torch.LongTensor(self.sizes):narrow(
                1, self.sizeIndex[k], self.sizeIndex[k + 1] - self.sizeIndex[k]
            ):copy(torch.LongTensor(v:size()))
        end
        self.numElements[1] = self.numElements[1] + 1
    end

    function mt:__len()
        assert(self)
        return self.numElements[1]
    end
    -- __len requires Lua version >= 5.2
    mt.len = mt.__len

    function mt:__write(f)
        assert(self)
        self:compress()
        f:writeLong(self.numElements[1])
        f:writeObject(self.sizeIndex)
        f:writeObject(self.sizes)
        f:writeObject(self.pointers)
        f:writeObject(self.content)
    end

    function mt:__read(f)
        assert(self)
        self.numElements[1] = f:readLong()
        self.sizeIndex = f:readObject()
        self.sizes     = f:readObject()
        self.pointers  = f:readObject()
        self.content   = f:readObject()
    end

    function mt:__tostring()
        assert(self)
        return(torch.typename(self))
    end

    function mt:__pairs()
        assert(self)
        local k = 0
        return function()
            k = k + 1
            if k <= self.numElements[1] then return k, self[k] end
        end
    end

    function mt:compress()
        assert(self)
        self:resize(self.numElements[1])
    end

    function mt:resize(newSize)
        assert(self)
        assert(newSize >= 0)

        -- compute new size of storages:
        self.numElements[1] = math.min(self.numElements[1], newSize)
        local contentSize = (self.numElements[1] == 0) and 0
            or self.pointers[self.numElements[1] + 1] - 1
        local sizeIndexSize = (self.numElements[1] == 0) and 0
            or self.sizeIndex[self.numElements[1] + 1] - 1

        -- resize storages:
        self.pointers:resize(self.numElements[1] + 1)
        self.sizeIndex:resize(self.numElements[1] + 1)
        self.sizes:resize(sizeIndexSize)
        self.content:resize(contentSize)
    end

    -- register Torch type:
    mt.__version = 0
    mt.__typename = vectorname
    mt.__factory = function(file) return mt.__new() end
    torch.metatype(vectorname, mt)
    return mt.__new
end

-- generate constructors for all Tensor types:
local M = {}
local typenames = {'Char', 'Byte', 'Short', 'Int', 'Long', 'Float', 'Double'}
for _,val in pairs(typenames) do
    M['new_' .. val:lower()] = defineAtomicTensorVector(val)
end

-- return module:
return M
