/*
 * Copyright 2020 NXP
 * Copyright 2021 ACRIOS Systems s.r.o.
 * All rights reserved.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "erpc_manually_constructed.hpp"
#include "erpc_transport_setup.h"
#include "erpc_usb_cdc_transport.hpp"

using namespace erpc;

////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////

ERPC_MANUALLY_CONSTRUCTED(UsbCdcTransport, s_usb_transport);

////////////////////////////////////////////////////////////////////////////////
// Code
////////////////////////////////////////////////////////////////////////////////

erpc_transport_t erpc_transport_usb_cdc_init(usb_handle_t handle)
{
    erpc_transport_t transport;

    s_usb_transport.construct(handle);
    if (s_usb_transport->init() == kErpcStatus_Success)
    {
        transport = reinterpret_cast<erpc_transport_t>(s_usb_transport.get());
    }
    else
    {
        transport = NULL;
    }

    return transport;
}
