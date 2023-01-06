module MappedBuffers

export MappedBuffer

using Mmap
using TranscodingStreams, CodecBzip2, CodecZlib, CodecZstd, CodecXz

# Size of chunks for transferring data. Should be a multiple of the memory page
# size (usually 4k). Even though the size of internal buffers in Julia i/o
# streams should warrant efficiency, having not too small chunks reduces the
# number of calls to read/write.
const CHUNK_SIZE = 4096*256

# Optional i/o stream.
const OptionalIO = Union{Nothing,IO}

@enum AccessMode::Cint begin
    CLOSED     = 0
    READ_ONLY  = 1
    WRITE_ONLY = 2
    READ_WRITE = 3
end

AccessMode(io::IOStream) =
    AccessMode((isreadable(io) ? 1 : 0)|(iswritable(io) ? 2 : 0))

"""
    buf = MappedBuffer(mode; kwds...)
    buf = MappedBuffer(; mode, kwds...)

yield a byte buffer `buf` whose contents is a memory-mapped region of a file.
The only mandatory argument/keyword is the access `mode` which is one of: `:r`
(read), `:w` (write), or `:rw` (read-write). All other settings are specified
by keywords. The returned object behaves as an abstract vector of bytes. Call
`resize!(buf,n)` to set the size of the buffer to `n` bytes or `fill!(buf)` to
set the buffer with the contents of the mapped file.


## Mapped File

Keyword `file` may be set with the path of the mapped file or an i/o stream to
the mapped file. If unspecified, a temporary file will be created. The
directory for the mapped file may be specified with keyword `dir`. If a
temporary mapped file is created, the default directory is given by
`tempdir()`. If `file` is specified with a relative path and `dir` is
specified, the path of the mapped file is `abspath(joinpath(dir,file))`. If
`file` is specified with an absolute path, `dir` must not be specified.
Finally, `file` may be specified as a tuple `(path,io)` with `path` the
absolute path to the mapped file and `io` a stream open to this file. This is
to allow for directly using the output of `mktemp()`.

Keyword `delete_file` may be used to specify whether to delete the mapped file
when the mapped buffer is closed. By default, `delete_file` is `true` if the
mapped file is a temporary file and `false` otherwise. Deleting the mapped file
is only performed if `close(buf)` is called. Nothing is done if the mapped
buffer is garbage collected before being closed or if `delete_file` is `false`.


## Input and output streams

A mapped buffer may be associated with an input stream and with an output
stream. This may be exploited to deal with compressed data. Keyword `input` may
be set with a filename or with a readable stream to specify an associated input
stream. Likewise, keyword `output` may be set with a filename or with a
writable stream to specify an associated output stream.

If `input` (resp. `output`) is specified as a stream, keyword `close_input`
(resp. `close_output`) may be set to `false` to not close this stream on
closing of the mapped buffer.

If an associated stream is specified by its filename, an i/o stream is
automacally open to access the file with automatic compression for an output
stream based on the filename extension and with automatic decompression for an
input stream based on the contents of the file.

"""
mutable struct MappedBuffer{I<:OptionalIO,
                            O<:OptionalIO} <: DenseVector{UInt8}
    input_bytes::Int # number of raw bytes read from input
    output_bytes::Int # number of raw bytes written to output
    mode::AccessMode
    delete_file::Bool # delete mapped file on close?
    close_input::Bool
    close_output::Bool
    array::Vector{UInt8} # memory mapped array reflecting the mapped file contents
    stream::IOStream # stream to mapped file
    path::String # normalized absolute path to mapped file
    input::I # associated input stream or `nothing`
    output::O # associated output stream or `nothing`

    # The following inner constructor is to prevent building a totally unusable
    # object. Simple default settings are chosen and may be further tuned by
    # the caller.
    function MappedBuffer{I,O}(mode::AccessMode,
                               delete::Bool,
                               stream::IOStream,
                               path::AbstractString,
                               input::I,
                               output::O) where {I<:OptionalIO,
                                                 O<:OptionalIO}
        return new{I,O}(0, 0, mode, delete, !isnothing(input), !isnothing(output),
                        UInt8[], stream, path, input, output)
    end
end

AccessMode(B::MappedBuffer) = getfield(B, :mode)

# NOTE: Making the mode the only required argument is simple mean to avoid ambiguities.
MappedBuffer(; mode::Symbol, kwds...) = MappedBuffer(mode; kwds...)

function MappedBuffer(mode::Symbol;
                      dir::Union{Nothing,AbstractString} = nothing,
                      file::Union{Nothing,IOStream,AbstractString,
                                  Tuple{AbstractString,IOStream}} = nothing,
                      delete_file::Bool = isnothing(file),
                      input = nothing,
                      close_input::Bool = !isnothing(input),
                      output = nothing,
                      close_output::Bool = !isnothing(output))

    # Check mandatory mode argument/keyword.
    if mode === :r
        mode = READ_ONLY
        isnothing(output) || throw(ArgumentError("no associated output is allowed in read-only mode"))
        isnothing(input) && isnothing(file) && throw(ArgumentError("no input specified in read-only mode"))
    elseif mode === :w
        mode = WRITE_ONLY
        isnothing(input) || throw(ArgumentError("no associated input is allowed in write-only mode"))
    elseif mode === :rw
        mode = READ_WRITE
    else
        throw(ArgumentError("invalid mode ($mode), should be `:r`, `:w`, or `:rw`"))
    end

    # Deal with mapped file.
    if isnothing(file)
        # Create a temporary file.
        if isnothing(dir)
            path, stream = mktemp(; cleanup=true)
        else
            path, stream = mktemp(dir; cleanup=true)
        end
    elseif file isa AbstractString
        # Open file.
        if isnothing(dir)
            path = abspath(file)
        elseif isabspath(file)
            throw(ArgumentError("absolute mapped file path must not specified with `dir` keyword"))
        else
            path = abspath(joinpath(dir, file))
        end
        stream = open(path, (mode == READ_ONLY ? "r" : mode == WRITE_ONLY ? "w+" : "r+"))
    else
        isnothing(dir) || throw(ArgumentError("`dir` keyword must be nothing in this case"))
        if file isa IOStream
            stream = file
            path = filename(stream)
        else # must be Tuple{String,IOStream}
            path, stream = file
        end
        isreadable(stream) || throw(ArgumentError("stream to mapped file must be readable"))
        mode == READ_ONLY || iswritable(stream) || throw(ArgumentError("stream to mapped file must be writable"))
    end
    return build(MappedBuffer, mode, delete_file, stream, String(path),
                 open_input(input), auto_close(input, close_input),
                 open_output(output), auto_close(output, close_output))
end

# Private type-stable constructor.
function build(::Type{MappedBuffer},
               mode::AccessMode,
               delete::Bool,
               stream::IOStream,
               path::String,
               input::I, close_input::Bool,
               output::O, close_output::Bool) where {I,O}
    try
        B = MappedBuffer{I,O}(mode, delete, stream, path, input, output)
        if !close_input
            B.close_input = false
        end
        if !close_output
            B.close_output = false
        end
        return B
    catch err
        # Close associated streams on error.
        close_input && !isnothing(input) && isopen(input) && close(input)
        close_output && !isnothing(output) && isopen(output) && close(output)
        isopen(file) && close(file)
        throw(err)
    end
end

# Helper to open input stream, if specified by filename, automatically using
# suitable decompression codec according to contents.
open_input(input) = input
function open_input(filename::AbstractString)
    io = open(filename, "r")
    codec = guess_codec(io)
    if codec === :gzip
        return GzipDecompressorStream(io)
    elseif codec === :bzip2
        return Bzip2DecompressorStream(io)
    elseif codec === :xz
        return XzDecompressorStream(io)
    elseif codec === :zlib
        return ZlibDecompressorStream(io)
    elseif codec === :zstd
        return ZstdDecompressorStream(io)
    else
        return io
    end
end

# Helper to open output stream, if specified by filename, automatically using
# suitable compression codec according to extension.
open_output(output) = output
function open_output(filename::AbstractString)
    io = open(filename, "w")
    codec = guess_codec(filename; read=false)
    if codec === :gzip
        return GzipCompressorStream(io)
    elseif codec === :bzip2
        return Bzip2CompressorStream(io)
    elseif codec === :xz
        return XzCompressorStream(io)
    elseif codec === :zlib
        return ZlibCompressorStream(io)
    elseif codec === :zstd
        return ZstdCompressorStream(io)
    else
        return io
    end
end

# Always close a stream that we open ourself (i.e., specified by its name),
# never close a missing stream, use caller choice otherwise.
auto_close(io::AbstractString, close::Bool) = true
auto_close(io::Nothing,        close::Bool) = false
auto_close(io::IO,             close::Bool) = close

# Implement do-block syntax.
MappedBuffer(f::Function; mode::Symbol, kwds...) = MappedBuffer(f, mode; kwds...)
function MappedBuffer(f::Function, mode::Symbol; kwds...)
    A = MappedBuffer(mode; kwds...)
    try
        return f(A)
    finally
        close(A)
    end
end

function Base.resize!(B::MappedBuffer, n::Integer)
    n = Int(n)::Int
    if length(B) != n
        isopen(B) || error("cannot resize closed mapped buffer")
        _resize!(B, n)
        length(B) == n || error(length(B) < n ?
            "short file or insufficient disk space" :
            "cannot reduce file size below number of written bytes")
    end
    return B
end

"""
    MappedBuffers.try_resize!(B, n) -> B

attempts to resize the mapped buffer `B` to `n` bytes. Compared to
`resize!(B,n)`, no exception is thrown if resizing `B` to `n` bytes is not
possible.

"""
function try_resize!(B::MappedBuffer, n::Integer)
    n = Int(n)::Int
    if length(B) != n && isopen(B)
        _resize!(B, n)
    end
    return B
end

# This private function must only be called if the mapped buffer is open and n
# is different from the actual size.
#
# There are several methods to determine the size of an IOStream:
# position(seekend(io)) or filesize(io). However none of these is fast:
# `seekend`, `position`, and `filesize` respectively take about 500ns, 30ns,
# and 1µs. We therefore keep track of the file size/position ourself.
#
# Getting the length of an array is extremely fast, so we use this to keep
# track of the size.
function _resize!(B::MappedBuffer, n::Int)
    writable = iswritable(B)
    if isnothing(B.input)
        # No associated input stream.
        maxlen = writable ? typemax(Int) : filesize(B.stream)
    else
        # Shall we transfer bytes from the input stream to the mapped file?
        if n > B.input_bytes && !eof(B.input)
            # For efficiency, reading is done by chunks so more bytes than
            # strictly needed may be transferred.
            seek(B.stream, B.input_bytes) # FIXME: not needed in principle
            chunk = Vector{UInt8}(undef, CHUNK_SIZE)
            while true
                nread = readbytes!(B.input, chunk)
                if nread == length(chunk)
                    write(B.stream, chunk)
                elseif nread > 0
                    write(B.stream, view(chunk, 1:nread))
                end
                B.input_bytes += nread
                if B.input_bytes ≥ n || nread < length(chunk)
                    # Enough bytes read or short input stream.
                    break
                end
            end
            flush(B.stream)
        end
        maxlen = writable ? typemax(Int) : B.input_bytes
    end
    minlen = B.output_bytes
    newlen = clamp(n, minlen, maxlen)
    oldlen = length(B)
    if newlen != oldlen
        if newlen ≤ 0
            B.array = UInt8[]
        else
            # If the mapped buffer is writable, make sure to synchronize the
            # contents of the mapped file with that of the buffer before
            # remapping.
            writable && Mmap.sync!(B.array)
            B.array = Mmap.mmap(B.stream, Vector{UInt8}, (newlen,), 0;
                                grow = writable, shared = writable)
        end
    end
    return nothing
end

"""
    fill!(B::MappedBuffer) -> B

fills mapped buffer `B` with unveiled bytes from mapped file or from associated
input stream.

"""
function Base.fill!(B::MappedBuffer)
    if isreadable(B)
        try_resize!(B, isnothing(B.input) ? filesize(B.stream) : typemax(Int))
    end
    return B
end

function Base.flush(B::MappedBuffer)
    if iswritable(B)
        len = length(B)
        len > 0 && Mmap.sync!(B.array)
        if !isnothing(B.output) && len > B.output_bytes
            # Output is not the memory mapped file and there are pending bytes.
            write(B.output, view(B.array, B.output_bytes + 1 : len))
            B.output_bytes = len
            flush(B.output)
        end
    end
    return nothing
end

Base.isopen(B::MappedBuffer) = (AccessMode(B) != CLOSED)
Base.isreadable(B::MappedBuffer) = (AccessMode(B) == READ_ONLY)|(AccessMode(B) == READ_WRITE)
Base.iswritable(B::MappedBuffer) = (AccessMode(B) == WRITE_ONLY)|(AccessMode(B) == READ_WRITE)
Base.isreadonly(B::MappedBuffer) = (AccessMode(B) == READ_ONLY)

function Base.close(B::MappedBuffer)
    if isopen(B)
        flush(B)
        final_size = sizeof(B)
        B.array = UInt8[]
        B.mode = CLOSED # never close twice
        if B.close_output && isopen(B.output)
            close(B.output)
        end
        if B.close_input && isopen(B.input)
            close(B.input)
        end
        if isopen(B.stream)
            close(B.stream)
        end
        if B.delete_file
            # Remove memory mapped file.
            rm(filename(B))
        end
    end
    return nothing
end

# Abstract array API.
Base.length(B::MappedBuffer) = length(B.array)
Base.size(B::MappedBuffer) = (length(B),)
Base.axes(B::MappedBuffer) = (Base.OneTo(length(B)),)
Base.IndexStyle(::Type{<:MappedBuffer}) = IndexLinear()
@inline Base.checkbounds(B::MappedBuffer, i::Int) =
    1 ≤ i ≤ length(B) || throw(BoundsError(B, i))
@inline function Base.getindex(B::MappedBuffer, i::Int)
    @boundscheck checkbounds(B, i)
    @inbounds r = getindex(B.array, i)
    return r
end
@inline function Base.setindex!(B::MappedBuffer, x, i::Int)
    # NOTE: If mapped file is not open for writing, attempting to set a value
    #       will throw a ReadOnlyMemoryError so there are no reasons to
    #       consider this specific case.
    @boundscheck checkbounds(B, i)
    @inbounds setindex!(B.array, x, i)
    return B
end

# Extend other base methods.
Base.sizeof(B::MappedBuffer) = sizeof(B.array)
Base.cconvert(::Type{Ptr{T}}, B::MappedBuffer) where {T} = B.array
Base.unsafe_convert(::Type{Ptr{T}}, B::MappedBuffer) where {T} =
    Base.unsafe_convert(Ptr{T}, B.array)

"""
    pathof(B::MappedBuffer) -> str

yields the name of the file to which `B` is mapped to.

"""
Base.pathof(B::MappedBuffer) = filename(B)

"""
    filesize(B::MappedBuffer) -> n

yields the size (in bytes) of the file to which `B` is mapped to.

"""
Base.filesize(B::MappedBuffer) = filesize(B.stream)

"""
    truncate(B::MappedBuffer)

truncates the file to which `B` is mapped to the current size of `B`. The
mapped buffer must not have been closed.

Truncating may be used to ensure that the mapped file reflects the final
contents of the mapped buffer.  For example:

    A = MappedBuffer(:rw, file="some_existing_file.dat", delete_file=false)
    resize!(A, 100) # set the size of the mapped part
    ... # set values in A
    truncate(A) # truncate the mapped file to have this size
    close(A)

"""
Base.truncate(B::MappedBuffer) = truncate(B.stream, length(B))

"""
    MappedBuffers.filename(io::IOStream)
    MappedBuffers.filename(B::MappedBuffer)

yield the file name of the stream `io` or of the file mapped by `B`.

"""
filename(B::MappedBuffer) = getfield(B, :path)
function filename(io::IOStream)
    name = io.name
    i = firstindex(name)
    j = lastindex(name)
    if j > i && (@inbounds name[j]) == '>'
        if startswith(name, "<file ")
            return String(SubString(name, nextind(name, i, 6), prevind(name, j)))
        end
        @static if Sys.islinux()
            if startswith(name, "<fd ")
                fd = tryparse(Int, SubString(name, nextind(name, i, 4), prevind(name, j)))
                if !isnothing(fd)
                    link = "/proc/self/fd/$fd"
                    isfile(link)
                    return String(readlink(link))
                end
            end
        end
    end
    error("cannot guess file for IOStream($name)")
end

"""
    MappedBuffers.guess_codec(file; read=isfile(filename))
    MappedBuffers.guess_codec(magic, n=length(magic))

yield the compression format of a file given its name, a stream, or a buffer
of `n` bytes read from the file. Returns one of: `:gzip`, `:bzip2`, `:xz`,
`:zlib`, `:zstd`, or `:other`.

If `file` is a stream, the compression format is determined by reading a few
bytes from the stream. The stream position is restored to its initial value, so
the stream must be seekable.

If `file` is a filename, keyword `read` may be used to specify whether the
compression format should be determined from the first bytes read from the
file; otherwise, the file extension is used to guess the compression.

"""
function guess_codec(io::IO)
    magic = Vector{UInt8}(undef, 4)
    pos = position(io)
    n = readbytes!(io, magic)
    n == 0 || seek(io, pos)
    return guess_codec(magic, n)
end

function guess_codec(magic::AbstractVector{UInt8}, n::Integer = length(magic))
    if n ≥ 3 && magic[1] == 0x1F && magic[2] == 0x8B && magic[3] == 0x08
        # GZIP with the DEFLATE method.
        return :gzip
    elseif n ≥ 3 && magic[1] == 0x42 && magic[2] == 0x5A && magic[3] == 0x68 # BZh
        return :bzip2
    elseif n ≥ 3 && magic[1] == 0xFD && magic[2] == 0x37 && magic[3] == 0x7A
        return :xz
    elseif n ≥ 4 && magic[1] == 0x28 && magic[2] == 0xb5 && magic[3] == 0x2f && magic[4] == 0xfd
        return :zstd
    elseif n ≥ 2 && magic[1] == 0x78 && magic[2] ∈ (0x01, 0x5E, 0x9C, 0xDA)
        # The above matches the most common cases. For more details, see
        # https://stackoverflow.com/questions/9050260/what-does-a-zlib-header-look-like
        return :zlib
    else
        return :other
    end
end

guess_codec(filename::AbstractString; read::Bool = isfile(filename)) =
    read ? guess_codec_from_contents(filename) : guess_codec_from_extension(filename)

function guess_codec_from_contents(filename::AbstractString)
    magic = Vector{UInt8}(undef, 4)
    n = open(filename, "r") do io
        readbytes!(io, magic)
    end
    return guess_codec(magic, n)
end

function guess_codec_from_extension(filename::AbstractString)
    i = findlast('.', filename)
    if !isnothing(i)
        ext = SubString(filename, i, lastindex(filename))
        for (sfx, codec) in ((".gz",  :gzip),
                             (".bz2", :bzip2),
                             (".xz",  :xz),
                             (".zst", :zstd))
            ext == sfx && return codec
        end
    end
    return :other
end

end # module
