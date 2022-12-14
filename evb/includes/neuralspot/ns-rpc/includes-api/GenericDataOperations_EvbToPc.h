/*
 * Generated by erpcgen 1.9.1 on Fri Sep  9 09:53:34 2022.
 *
 * AUTOGENERATED - DO NOT EDIT
 */

#if !defined(_GenericDataOperations_EvbToPc_h_)
    #define _GenericDataOperations_EvbToPc_h_

    #include "erpc_version.h"
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>

    #if 10901 != ERPC_VERSION_NUMBER
        #error "The generated shim code version is different to the rest of eRPC code."
    #endif

    #if !defined(ERPC_TYPE_DEFINITIONS)
        #define ERPC_TYPE_DEFINITIONS

// Enumerators data types declarations
typedef enum status {
    ns_rpc_data_success = 0,
    ns_rpc_data_failure = 1,
    ns_rpc_data_blockTooLarge = 2
} status;

typedef enum dataType {
    uint8_e = 0,
    uint16_e = 1,
    uint32_e = 2,
    int8_e = 3,
    int16_e = 4,
    int32_e = 5,
    float32_e = 6,
    float64_e = 7
} dataType;

typedef enum command {
    generic_cmd = 0,   //!< The sender doesn't have an opinion about what to do with data
    visualize_cmd = 1, //!< Block intended for visualization
    infer_cmd = 2,     //!< Compute inference for block
    extract_cmd = 3,   //!< Compute feature from block
    write_cmd = 4,     //!< Block intended for writing to a file
    read = 5           //!< Fetch block from a file
} command;

// Aliases data types declarations
typedef struct binary_t binary_t;
typedef struct dataBlock dataBlock;

// Structures/unions data types declarations
struct binary_t {
    uint8_t *data;
    uint32_t dataLength;
};

struct dataBlock {
    uint32_t length;   //!< In bytes
    dataType dType;    //!< Type of data in block
    char *description; //!< Textual description/metadata
    command cmd;       //!< Suggestion of what to do with data
    binary_t buffer;   //!< The data
};

    #endif // ERPC_TYPE_DEFINITIONS

/*! @brief evb_to_pc identifiers */
enum _evb_to_pc_ids {
    kevb_to_pc_service_id = 1,
    kevb_to_pc_ns_rpc_data_sendBlockToPC_id = 1,
    kevb_to_pc_ns_rpc_data_fetchBlockFromPC_id = 2,
    kevb_to_pc_ns_rpc_data_computeOnPC_id = 3,
    kevb_to_pc_ns_rpc_data_remotePrintOnPC_id = 4,
};

    #if defined(__cplusplus)
extern "C" {
    #endif

//! @name evb_to_pc
//@{
status
ns_rpc_data_sendBlockToPC(const dataBlock *block);

status
ns_rpc_data_fetchBlockFromPC(dataBlock *block);

status
ns_rpc_data_computeOnPC(const dataBlock *in_block, dataBlock *result_block);

status
ns_rpc_data_remotePrintOnPC(const char *msg);
//@}

    #if defined(__cplusplus)
}
    #endif

#endif // _GenericDataOperations_EvbToPc_h_
