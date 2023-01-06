module TestingMappedBuffers

using MappedBuffers
using TranscodingStreams, CodecBzip2, CodecZlib, CodecZstd, CodecXz
using Test

data = rand(UInt8, 234567)
path1, io1 = mktemp(;cleanup=true)
path2, io2 = mktemp(;cleanup=true)
path3, io3 = mktemp(;cleanup=true)

@testset "MappedBuffers.jl" begin
    @testset "Utilities" begin
        @test MappedBuffers.AccessMode(io1) == MappedBuffers.READ_WRITE
        if Sys.islinux()
            @test MappedBuffers.filename(io1) == path1
        else
            @test_throws ErrorException MappedBuffers.filename(io1)
        end
        let guess_codec = MappedBuffers.guess_codec
            @test guess_codec("a/b.c.gr", read=false) === :other
            @test guess_codec("a/b.c.gz", read=false) === :gzip
            @test guess_codec("a/b.c.xz", read=false) === :xz
            @test guess_codec("a/b.c.bz2", read=false) === :bzip2
            @test guess_codec("a/b.c.zst", read=false) === :zstd
            @test guess_codec(UInt8[]) === :other
        end
    end
    @testset "Writing raw data" begin
        @test_throws ArgumentError MappedBuffer(:invalid_mode)
        MappedBuffer(mode=:w, file=(path1,io1)) do A
            @test A isa DenseVector{UInt8}
            @test isreadable(A) == false
            @test iswritable(A) == true
            @test isreadonly(A) == false
            @test isopen(A) == true
            @test IndexStyle(A) === IndexLinear()
            @test A.delete_file == false
            @test A.input_bytes == 0
            @test A.output_bytes == 0
            @test length(A) == sizeof(A) == 0
            @test resize!(A, sizeof(data)) === A
            @test length(A) == sizeof(A) == sizeof(data)
            @test size(A) == (length(A),)
            @test axes(A) == (1:length(A),)
            @test all(iszero, A)
            # Exercise pointer, Base.unsafe_convert, Base.cconvert, ... (NOTE:
            # @ccall does not exist before Julia 1.5)
            @test ccall(:memcpy, Ptr{UInt8}, (Ptr{UInt8}, Ptr{UInt8}, Csize_t),
                        A, data, sizeof(A)) === pointer(A)
            @test A == data
            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), A, 0, sizeof(A))
            @test all(iszero, A)
            for i in eachindex(A,data)
                A[i] = data[i]
            end
            @test_throws BoundsError A[0]
            @test_throws BoundsError A[end+1]
            flush(A)
            @test read(pathof(A)) == data
            # Resizing to 0 and then to original size preserve contents.
            @test length(resize!(A,0)) == 0
            flush(A)
            @test sizeof(resize!(A, sizeof(data))) == sizeof(data)
            flush(A)
            @test read(pathof(A)) == data
        end
        @test filesize(path1) == sizeof(data)
        @test read(path1) == data
    end
    @testset "Reading raw data" begin
        for fill in (false, true)
            kwds = (fill ? () : (fill=false,)) # default is to fill
            MappedBuffer(:r; file=open(path1), kwds...) do A
                @test A isa DenseVector{UInt8}
                @test isreadable(A) == true
                @test iswritable(A) == false
                @test isreadonly(A) == true
                @test isopen(A) == true
                @test IndexStyle(A) === IndexLinear()
                @test A.delete_file == false
                @test A.input_bytes == 0
                @test A.output_bytes == 0
                if !fill
                    @test length(A) == sizeof(A) == 0
                    @test fill!(A) === A # manually map to all contents
                end
                @test length(A) == sizeof(A) == sizeof(data)
                @test A.input_bytes == 0
                @test A.output_bytes == 0
                @test A == data
                @test filesize(A) == filesize(pathof(A))
            end
        end
    end
    @testset "Updating raw data" begin
        for fill in (false, true)
            kwds = (fill ? () : (fill=false,)) # default is to fill
            MappedBuffer(:rw; file=path1, kwds...) do A
                @test A isa DenseVector{UInt8}
                @test isreadable(A) == true
                @test iswritable(A) == true
                @test isreadonly(A) == false
                @test isopen(A) == true
                @test IndexStyle(A) === IndexLinear()
                @test A.delete_file == false
                @test A.input_bytes == 0
                @test A.output_bytes == 0
                if !fill
                    @test length(A) == sizeof(A) == 0
                    @test fill!(A) === A # map to all contents
                end
                @test length(A) == sizeof(A) == sizeof(data)
                @test A.input_bytes == 0
                @test A.output_bytes == 0
                @test A == data
                r = firstindex(A):2:lastindex(A)
                A[r] = .~(A[r]) # XOR some bytes
                flush(A)
                B = read(pathof(A))
                @test B == A
                @test B != data
                n = length(A)
                m = 17
                resize!(A, n + m)
                @test length(A) == n + m
                @test all(iszero, view(A, n + 1 : n + m))
                A[n+1:n+m] .= 0x31
                flush(A)
                @test read(pathof(A)) == A
                A[r] = .~(A[r]) # reverse XOR of bytes
                resize!(A, n) # restore old size
                flush(A)
                B = read(pathof(A))
                @test length(B) == n + m
                @test B[1:n] == A
                @test all(x -> x == 0x31, B[n+1:end])
                truncate(A)
                @test read(pathof(A)) == A
            end
        end
        @test read(path1) == data
    end
    @testset "Reading compressed data ($alg)" for (alg, enc, dec) in (
        (:bzip2, Bzip2Compressor, Bzip2Decompressor),
        (:gzip,  GzipCompressor,  GzipDecompressor),
        (:xz,    XzCompressor,    XzDecompressor),
        (:zlib,  ZlibCompressor,  ZlibDecompressor),
        (:zstd,  ZstdCompressor,  ZstdDecompressor))
        s = TranscodingStream{enc}(open(path1, "w"))
        write(s, data)
        close(s)
        let guess_codec = MappedBuffers.guess_codec
            @test guess_codec(path1, read=true) === alg
            @test open(path1) do io
                guess_codec(io)
            end === alg
        end
        s = TranscodingStream{dec}(open(path1, "r"))
        temp = read(s)
        close(s)
        @test temp == data
        for fill in (false, true)
            kwds = (fill ? () : (fill=false,)) # default is to fill
            A = MappedBuffer(:r; input=TranscodingStream{dec}(open(path1, "r")), kwds...)
            @test A isa DenseVector{UInt8}
            @test isopen(A) == true
            @test isreadable(A) == true
            @test iswritable(A) == false
            @test isreadonly(A) == true
            if !fill
                @test length(A) == sizeof(A) == 0
                @test A.input_bytes == 0
                @test A.output_bytes == 0
                # Read ~ 1/3rd of the input file.
                n = div(sizeof(data), 3)
                @test resize!(A, n) === A
                @test length(A) == sizeof(A) == n
                @test A.input_bytes ≥ n
                @test A == view(data, 1:n)
                # Read all remaining data.
                @test fill!(A) === A
            end
            @test length(A) == sizeof(A) == sizeof(data)
            @test A.input_bytes == sizeof(data)
            @test A.output_bytes == 0
            @test A == data
            close(A)
            @test isopen(A) == false
        end
    end
    @testset "Writing compressed data ($alg)" for (alg, enc, dec) in (
        (:bzip2, Bzip2Compressor, Bzip2Decompressor),
        (:gzip,  GzipCompressor,  GzipDecompressor),
        (:xz,    XzCompressor,    XzDecompressor),
        (:xlib,  ZlibCompressor,  ZlibDecompressor),
        (:zstd,  ZstdCompressor,  ZstdDecompressor))
        MappedBuffer(:w, output=TranscodingStream{enc}(open(path1, "w"))) do A
            @test A.delete_file == true
            @test A.input_bytes == 0
            @test A.output_bytes == 0
            @test length(A) == 0
            @test sizeof(A) == 0
            # Write ~ 1/3rd of the input file.
            n = div(sizeof(data), 3)
            @test resize!(A, n) === A
            @test length(A) == n
            @test A.input_bytes == 0
            @test A.output_bytes == 0
            A[:] = view(data, 1:n)
            flush(A)
            @test A.input_bytes == 0
            @test A.output_bytes == sizeof(A) == n
            # Write all remaining data.
            @test resize!(A, sizeof(data)) === A
            @test length(A) == sizeof(data)
            @test sizeof(A) == sizeof(data)
            @test A[1:n] == data[1:n]
            @test all(iszero, A[n+1:end])
            for i in n+1:length(A)
                A[i] = data[i]
            end
            @test A == data
            flush(A)
            @test A.output_bytes == sizeof(A) == sizeof(data)
        end
        # Check integrity.
        open(TranscodingStream{dec}, path1) do io
            @test read(io) == data
        end
        # Re-open the compressed file given its name to whether guessing the
        # codec type from contents works.
        MappedBuffer(:r, input=path1) do A
            @test A == data
        end
    end
end

end # module
