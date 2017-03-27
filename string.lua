--[[

This is a simple string vector implementation on top of Torch. It is
serializable so you can save it in a file, and it can be stored in tds.hash.

]]--


-- dependencies:
require 'torch'
local ffi = require 'ffi'
local _tds = require 'tds'
local elem = require 'tds.elem'

-- define C type:
ffi.cdef[[
    typedef struct {
      THCharStorage *content;
      THLongStorage *pointers;
      THLongStorage *contentSize;
      THLongStorage *numPointers;
    } THStringVector;

    void THCharStorage_free(THCharStorage *storage);
    void THLongStorage_free(THLongStorage *storage);

    void *malloc(size_t size);
    void free(void *ptr);
]]

-- register C-structure in tds:
local mt = {}
local function free_p(celem)    -- frees the C-structure
    celem = ffi.cast('THStringVector*', celem)
    ffi.C['THCharStorage_free'](celem.content)
    ffi.C['THLongStorage_free'](celem.pointers)
    ffi.C['THLongStorage_free'](celem.contentSize)
    ffi.C['THLongStorage_free'](celem.numPointers)
    ffi.C.free(celem)
end
free_p = ffi.cast('void (*)(void*)', free_p)
local function set_func(lelem)  -- sets the C-structure
    local celem = ffi.cast(
        'THStringVector*',
        ffi.C.malloc(ffi.sizeof('THStringVector'))
    )
    celem.content  = lelem.content:cdata()
    lelem.content:retain()
    celem.pointers = lelem.pointers:cdata()
    lelem.pointers:retain()
    celem.contentSize = lelem.contentSize:cdata()
    lelem.contentSize:retain()
    celem.numPointers = lelem.numPointers:cdata()
    lelem.numPointers:retain()
    return celem, free_p
end
local function get_func(celem)   -- gets the C-structure
    local celem = ffi.cast('THStringVector*', celem)
    local lelem = {}
    lelem.contentSize = torch.pushudata(celem.contentSize, 'torch.LongStorage')
    lelem.contentSize:retain()
    lelem.numPointers = torch.pushudata(celem.numPointers, 'torch.LongStorage')
    lelem.numPointers:retain()
    lelem.content  = torch.pushudata(celem.content,  'torch.CharStorage')
    lelem.content:retain()
    lelem.pointers = torch.pushudata(celem.pointers, 'torch.LongStorage')
    lelem.pointers:retain()
    setmetatable(lelem, mt)
    return lelem
end
elem.addctype(  -- adds type as a tds element
    'StringVector',
    free_p,
    set_func,
    get_func
)

-- define all functions in the metatable:
function mt.__new()
    local self = {}
    self.content = torch.CharStorage(1)
    self.pointers = torch.LongStorage(1)
    self.pointers[1] = 1
    self.contentSize = torch.LongStorage(1)
    self.numPointers = torch.LongStorage(1)
    self.contentSize[1] = 0  -- using a number does not play nice with tds.hash
    self.numPointers[1] = 0
    setmetatable(self, mt)
    return self
end

function mt:__index(k)
    assert(self)
    if type(k) == 'string' then return rawget(mt, k)    end
    if k <= 0 or k > self.numPointers[1] then error('index out of bounds') end
    local stringLength = self.pointers[k + 1] - self.pointers[k]
    return (stringLength == 0) and '' or ffi.string(
        torch.data(
            torch.CharTensor(self.content):narrow(1, self.pointers[k], 1)
        ),
        stringLength
    )
end

function mt:__newindex(k, v)
    assert(self)
    if k ~= self.numPointers[1] + 1 then error('only sequential writing') end
    if not v then error('removal not supported') end
    if type(v) ~= 'string' then error('can only insert strings') end

    while self.contentSize[1] + v:len() > self.content:size() do
        self.content:resize(self.content:size() * 2)
    end
    if k > (self.pointers:size() - 1) then
        self.pointers:resize(self.pointers:size() * 2)
    end

    self.pointers[k + 1] = self.pointers[k] + v:len()
    if v:len() > 0 then
        ffi.copy(
            torch.data(torch.CharTensor(self.content):narrow(
                1, self.pointers[k], self.pointers[k + 1] - self.pointers[k]
            )),
            v, v:len()
        )
    end
    self.numPointers[1] = self.numPointers[1] + 1
    self.contentSize[1] = self.contentSize[1] + v:len()
end

function mt:__len()
    assert(self)
    return self.numPointers[1]
end
-- __len requires Lua version >= 5.2
mt.len = mt.__len

function mt:__write(f)
    assert(self)
    self:compress()
    f:writeLong(self.numPointers[1])
    f:writeLong(self.contentSize[1])
    f:writeObject(self.pointers)
    f:writeObject(self.content)
end

function mt:__read(f)
    assert(self)
    self.numPointers[1] = f:readLong()
    self.contentSize[1] = f:readLong()
    self.pointers = f:readObject()
    self.content  = f:readObject()
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
        if k <= self.numPointers[1] then return k, self[k] end
    end
end

function mt:compress()
    assert(self)
    self:resize(self.numPointers[1])
end

function mt:resize(newSize)
    assert(self)
    assert(newSize >= 0)
    self.numPointers[1] = math.min(self.numPointers[1], newSize)
    self.contentSize[1] =
        math.min(self.contentSize[1], self.pointers[self.numPointers[1] + 1])
    self.content:resize(self.contentSize[1])
    self.pointers:resize(self.numPointers[1] + 1)
end

-- register Torch type:
mt.__version = 0
mt.__typename = 'StringVector'
mt.__factory = function(file) return mt.__new() end
torch.metatype('StringVector', mt)

-- return module:
return {new = mt.__new}
