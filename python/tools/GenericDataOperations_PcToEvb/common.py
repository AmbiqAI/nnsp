#
# Generated by erpcgen 1.9.1 on Thu Sep 15 09:19:22 2022.
#
# AUTOGENERATED - DO NOT EDIT
#


# Enumerators data types declarations
class status:
    ns_rpc_data_success = 0
    ns_rpc_data_failure = 1
    ns_rpc_data_blockTooLarge = 2


class dataType:
    uint8_e = 0
    uint16_e = 1
    uint32_e = 2
    int8_e = 3
    int16_e = 4
    int32_e = 5
    float32_e = 6
    float64_e = 7


class command:
    generic_cmd = 0  # The sender doesn't have an opinion about what to do with data
    visualize_cmd = 1  # Block intended for visualization
    infer_cmd = 2  # Compute inference for block
    extract_cmd = 3  # Compute feature from block
    write_cmd = 4  # Block intended for writing to a file
    read = 5  # Fetch block from a file


# Structures data types declarations
class dataBlock(object):
    def __init__(
        self, length=None, dType=None, description=None, cmd=None, buffer=None
    ):
        self.length = length  # uint32
        self.dType = dType  # dataType
        self.description = description  # string
        self.cmd = cmd  # command
        self.buffer = buffer  # binary

    def _read(self, codec):
        self.length = codec.read_uint32()
        self.dType = codec.read_uint32()
        self.description = codec.read_string()
        self.cmd = codec.read_uint32()
        self.buffer = codec.read_binary()
        return self

    def _write(self, codec):
        if self.length is None:
            raise ValueError("length is None")
        codec.write_uint32(self.length)
        if self.dType is None:
            raise ValueError("dType is None")
        codec.write_uint32(self.dType)
        if self.description is None:
            raise ValueError("description is None")
        codec.write_string(self.description)
        if self.cmd is None:
            raise ValueError("cmd is None")
        codec.write_uint32(self.cmd)
        if self.buffer is None:
            raise ValueError("buffer is None")
        codec.write_binary(self.buffer)

    def __str__(self):
        return "<%s@%x length=%s dType=%s description=%s cmd=%s buffer=%s>" % (
            self.__class__.__name__,
            id(self),
            self.length,
            self.dType,
            self.description,
            self.cmd,
            self.buffer,
        )

    def __repr__(self):
        return self.__str__()
