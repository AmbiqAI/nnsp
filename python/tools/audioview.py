"""_summary_

Returns:
    _type_: _description_
"""
import argparse
import sys
import wave
from multiprocessing import Process, Array, Lock
import erpc
import GenericDataOperations_EvbToPc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Define the RPC service handlers - one for each EVB-to-PC RPC function
FRAMES_TO_SHOW  = 500
SAMPLING_RATE   = 16000
HOP_SIZE        = 160

class DataServiceHandler:
    """
    Audio data: EVB->PC
    """
    def __init__(self, databuf, wavout, lock, shared_record):
        self.cyc_count      = 0
        self.wavefile       = None
        self.wavename       = wavout
        self.databuf        = databuf
        self.lock           = lock
        self.shared_record  = shared_record

    def wavefile_init(self, wavename):
        """
        wavefile initialization
        """
        # daytime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wavefile = wave.open(wavename, 'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        return wavefile

    def ns_rpc_data_sendBlockToPC(self, block): # pylint: disable=invalid-name
        """
        callback
        """
        self.lock.acquire()
        shared_record = self.shared_record[0]
        self.lock.release()
        if shared_record == 0:
            if self.wavefile:
                self.wavefile.close()
                self.wavefile = None
                print('Stop recording')
        else:
            if self.wavefile:
                self.wavefile.writeframesraw(block.buffer)
            else:
                print('Start recording')
                self.cyc_count = 0
                self.wavefile = self.wavefile_init(self.wavename)

            # Data is a 16 bit PCM sample
            self.lock.acquire()
            fdata = np.frombuffer(block.buffer, dtype=np.int16).copy() / 32768.0
            self.lock.release()
            start = self.cyc_count * HOP_SIZE
            if self.cyc_count == 0:
                np_databuf = np.zeros(FRAMES_TO_SHOW * HOP_SIZE)
                self.lock.acquire()
                self.databuf[0:] = np_databuf
                self.lock.release()
            self.lock.acquire()
            self.databuf[start:start+HOP_SIZE] = fdata
            self.lock.release()
            self.cyc_count = (self.cyc_count+1) % FRAMES_TO_SHOW

        sys.stdout.flush()

        return 0

class EvbDataClass:
    """
    Drawing the audio data from EVB
    """
    def __init__(self, databuf, lock, shared_record):
        self.databuf = databuf
        self.lock    = lock
        self.shared_record = shared_record
        secs2show = FRAMES_TO_SHOW * HOP_SIZE/SAMPLING_RATE
        self.xdata = np.arange(FRAMES_TO_SHOW * HOP_SIZE) / SAMPLING_RATE
        self.fig, self.ax_handle = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        self.lock.acquire()
        np_databuf = databuf[0:]
        self.lock.release()
        self.line_data, = self.ax_handle.plot(self.xdata, np_databuf, lw=0.5, color = 'blue')
        plt.ylim([-1.1,1.1])
        self.ax_handle.set_xlim((0, secs2show))
        self.ax_handle.set_xlabel('Time (Seconds)')
        plt.plot(
            [0, secs2show],
            [1, 1],
            color='black',
            lw=1)
        plt.plot(
            [0, secs2show],
            [-1, -1],
            color='black',
            lw=1)
        # making buttons
        def make_button(pos, name, callback_func):
            ax_button = plt.axes(pos)
            button = Button(
                        ax_button,
                        name,
                        color = 'w',
                        hovercolor = 'aliceblue')
            button.label.set_fontsize(16)
            button.on_clicked(callback_func)
            return button
        self.wavfile = None
        self.button_replay = make_button(
                            [0.35, 0.15, 0.14, 0.075],
                            'stop',
                            self.callback_recordstop)
        self.button_record = make_button(
                            [0.5, 0.15, 0.14, 0.075],
                            'record',
                            self.callback_recordstart)
        plt.show()

    def callback_recordstop(self, event):
        """
        for stop button
        """
        self.lock.acquire()
        self.shared_record[0] = 0
        self.lock.release()
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

    def callback_recordstart(self, event):
        """
        for record button
        """
        self.lock.acquire()
        shared_record = self.shared_record[0]
        self.lock.release()
        if shared_record == 0:
            self.lock.acquire()
            self.shared_record[0] = 1
            self.lock.release()
            while 1:
                self.lock.acquire()
                np_databuf = self.databuf[0:]
                self.lock.release()
                self.line_data.set_data(self.xdata, np_databuf)

                plt.pause(0.05)
                self.lock.acquire()
                shared_record = self.shared_record[0]
                self.lock.release()
                if shared_record == 0:
                    break
        if event.inaxes is not None:
            event.inaxes.figure.canvas.draw_idle()

def draw(databuf, lock, recording):
    """
    draw
    """
    EvbDataClass(databuf, lock, recording)

def rvd_evb(tty, baud, databuf, wavout, lock, shared_record):
    """
    EVB sends data to PC
    """
    transport_evb2pc = erpc.transport.SerialTransport(tty, int(baud))
    handler = DataServiceHandler(databuf, wavout, lock, shared_record)
    service = GenericDataOperations_EvbToPc.server.evb_to_pcService(handler)
    server = erpc.simple_server.SimpleServer(transport_evb2pc, erpc.basic_codec.BasicCodec)
    server.add_service(service)
    print("\r\nServer started - waiting for EVB to send an eRPC request")
    sys.stdout.flush()
    server.run()

def main(args):
    """
    main
    """
    lock = Lock()
    databuf = Array('d', FRAMES_TO_SHOW * HOP_SIZE)
    record_ind = Array('i', 1)
    record_ind[0] = 0
    proc_draw = Process(target=draw, args = (databuf,lock, record_ind))
    proc_main = Process(target=rvd_evb,
                        args = (args.tty,
                                args.baud,
                                databuf,
                                args.out,
                                lock,
                                record_ind))
    proc_draw.start()
    proc_main.start()
    proc_draw.join()
    proc_main.join()

if __name__ == "__main__":

    # parse cmd parameters
    argParser = argparse.ArgumentParser(description="NeuralSPOT GenericData RPC Demo")

    argParser.add_argument(
        "-w",
        "--tty",
        default="COM4", # "/dev/tty.usbmodem1234561"
        help="Serial device (default value is None)",
    )
    argParser.add_argument(
        "-B",
        "--baud",
        default="115200",
        help="Baud (default value is 115200)"
    )
    argParser.add_argument(
        "-o",
        "--out",
        default="audio.wav",
        help="File where data will be written (default is audio.wav",
    )

    main(argParser.parse_args())
