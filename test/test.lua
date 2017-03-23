-- dependencies:
require 'fb.luaunit'
require 'fbtorch'
local vector = require 'vector'

-- set up tester:
local tester = torch.Tester()
local test = {}

function test.tensorvector()

    -- fill and read tensorvector:
    local limit = 6
    local b = vector.tensor.new_double()
    for n = 1,limit do b[n] = n end
    for n = 1,limit do assert(b[n] == n) end
    for n = 1,limit do
        local i = math.random(limit)
        assert(b[i] == i)
    end

    -- check tensorvector pairs() functionality:
    local cnt = 0
    for key, val in pairs(b) do
        cnt = cnt + 1
        assert(key == cnt)
        assert(val == cnt)
    end
    assert(cnt == limit)

    -- checks __len() and typename:
    assert(#b == limit)
    assert(torch.typename(b) == 'DoubleVector')

    -- check read and write:
    local tmpName = os.tmpname()
    torch.save(tmpName, b)
    local bCopy = torch.load(tmpName)
    assert(torch.typename(b) == torch.typename(bCopy))
    assert(#b == #bCopy)
    for n = 1,#b do assert(b[n] == bCopy[n]) end

    -- check whether we can get the tensor:
    local bt = b:getTensor()
    assert(torch.typename(bt) == 'torch.DoubleTensor')
    for n = 1,limit do assert(bt[n] == b[n]) end

    -- check all constructors:
    local types = {'char', 'byte', 'short', 'int', 'long', 'float', 'double'}
    for _,val in pairs(types) do
        local c = vector.tensor['new_' .. val](1)
        c[1] = 12
        assert(c[1] == 12)
        local Type = val:sub(1, 1):upper() .. val:sub(2)
        assert(torch.typename(c) == Type .. 'Vector')
        assert(torch.typename(c:getTensor()) == 'torch.' .. Type .. 'Tensor')
    end

    -- check compress and resize:
    b:compress()
    assert(#b == limit)
    for n = 1,limit do assert(b[n] == n) end
    local smallLimit = 4
    b:resize(smallLimit)
    assert(#b == smallLimit)
    for n = 1,smallLimit do assert(b[n] == n) end

    -- check tds.hash() compatibility:
    local tds = require 'tds'
    local tdsHash = tds.hash()
    for _,val in pairs(types) do
        tdsHash[#tdsHash + 1] = vector.tensor['new_' .. val](1)
        for n = 1,limit do tdsHash[#tdsHash][n] = n end
        for n = 1,limit do assert(tdsHash[#tdsHash][n] == n) end
    end

    -- check whether running other tensor functions works:
    bt = b:getTensor():exp()
    b:exp(); bt:exp()
    for n = 1,smallLimit do assert(bt[n] == b[n]) end
    b:log(); bt:log()
    for n = 1,smallLimit do assert(bt[n] == b[n]) end
    b:add(1); bt:add(1)
    for n = 1,smallLimit do assert(bt[n] == b[n]) end

    -- check whether tensor functions that allocate new memory work:
    assert(b:sum() == bt:sum())
    local bc  = torch.cumsum(b:getTensor(), 1)  -- this requires :getTensor()
    local btc = torch.cumsum(bt, 1)
    for n = 1,smallLimit do assert(bc[n] == btc[n]) end

    -- check whether you can insert tensors:
    local b = vector.tensor.new_double()
    local ind = 2
    local r = torch.rand(2, 2)
    b:insertTensor(ind, r)
    r = r:resize(r:nElement())
    for i = 1,r:nElement() do assert(r[i] == b[ind + i - 1]) end
end

function test.stringvector()

    -- make big list of strings:
    local limit = 6
    local strings = {}
    for n = 1,limit do strings[n] = os.tmpname() end
    strings[2] = ''  -- make sure we can deal with empty strings

    -- check inserting and accessing:
    local b = vector.string.new()
    for key, val in pairs(strings) do b[key] = val end
    for n = 1,#b do assert(b[n] == strings[n]) end
    for n = 1,#b do
        local i = math.random(#b)
        assert(b[i] == strings[i])
    end

    -- check pairs() functionality:
    local cnt = 0
    for key, val in pairs(b) do
        cnt = cnt + 1
        assert(key == cnt)
        assert(val == strings[key])
    end
    assert(cnt == #strings)

    -- checks __len() and typename:
    assert(#b == #strings)
    assert(torch.typename(b) == 'StringVector')

    -- check read and write:
    local tmpName = os.tmpname()
    torch.save(tmpName, b)
    local bCopy = torch.load(tmpName)
    assert(torch.typename(b) == torch.typename(bCopy))
    assert(#b == #bCopy)
    for n = 1,#b do assert(b[n] == bCopy[n]) end

    -- check compress and resize:
    b:compress()
    assert(#b == limit)
    for n = 1,limit do assert(b[n] == strings[n]) end
    local smallLimit = 4
    b:resize(smallLimit)
    assert(#b == smallLimit)
    for n = 1,smallLimit do assert(b[n] == strings[n]) end

    -- check tds.hash() compatibility:
    local tds = require 'tds'
    local tdsHash = tds.hash()
    for k = 1,3 do
        tdsHash[#tdsHash + 1] = vector.string.new()
        for key, val in pairs(strings) do
            tdsHash[#tdsHash][key] = val
        end
        for key, val in pairs(strings) do
            assert(tdsHash[#tdsHash][key] == val)
        end
    end
end

function test.atomicvector()
    local tds = require 'tds'
    local function tensorEqual(a, b)
        if a:nElement() == 0 and b:nElement() == 0 then return true end
        return torch.eq(a, b):all()
    end

    -- fill and read tensorvector:
    local limit = 6
    local a = tds.hash()
    local b = vector.atomic.new_double()
    for n = 1,limit - 1 do
        local size = torch.LongTensor(n):random(5)
        local tensor = torch.DoubleTensor(size:storage()):uniform()
        a[n] = tensor
        b[n] = tensor
    end
    a[limit] = torch.DoubleTensor()  -- test for empty tensor
    b[limit] = torch.DoubleTensor()
    for n = 1,limit do
        assert(tensorEqual(a[n], b[n]))
    end
    for n = 1,limit do
        local i = math.random(limit)
        assert(tensorEqual(a[i], b[i]))
    end

    -- check tensorvector pairs() functionality:
    local cnt = 0
    for key, val in pairs(b) do
        cnt = cnt + 1
        assert(key == cnt)
        assert(tensorEqual(a[cnt], val))
    end
    assert(cnt == limit)

    -- checks __len() and typename:
    assert(#b == limit)
    assert(torch.typename(b) == 'DoubleAtomicVector')

    -- check read and write:
    local tmpName = os.tmpname()
    torch.save(tmpName, b)
    local bCopy = torch.load(tmpName)
    assert(torch.typename(b) == torch.typename(bCopy))
    assert(#b == #bCopy)
    for n = 1,#b do assert(tensorEqual(b[n], bCopy[n])) end

    -- check all constructors:
    local types = {'char', 'byte', 'short', 'int', 'long', 'float', 'double'}
    for _,val in pairs(types) do
        local c = vector.atomic['new_' .. val](1)
        local Type = val:sub(1, 1):upper() .. val:sub(2)
        c[1] = torch[Type .. 'Tensor'](1):fill(12)
        assert(c[1][1] == 12)
        assert(torch.typename(c) == Type .. 'AtomicVector')
    end

    -- check compress and resize:
    b:compress()
    assert(#b == limit)
    for n = 1,limit do assert(tensorEqual(a[n], b[n])) end
    local smallLimit = 4
    b:resize(smallLimit)
    assert(#b == smallLimit)
    for n = 1,smallLimit do assert(tensorEqual(a[n], b[n])) end

    -- check tds.hash() compatibility:
    local tds = require 'tds'
    local tdsHash = tds.hash()
    for _,val in pairs(types) do
        local Type = val:sub(1, 1):upper() .. val:sub(2)
        tdsHash[#tdsHash + 1] = vector.atomic['new_' .. val](1)
        for n = 1,limit do
            local size = torch.LongTensor(n):random(5)
            local tensor = torch[Type .. 'Tensor'](size:storage()):fill(n)
            tdsHash[#tdsHash][n] = tensor
            assert(tensorEqual(tdsHash[#tdsHash][n], tensor))
        end
    end
end

-- run the unit tests:
torch.setnumthreads(1)
tester:add(test)
tester:run()
