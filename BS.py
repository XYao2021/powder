import argparse
import uhd
import time
from utils import *

class Radio():
    RX_CLEAR_COUNT = 1000
    LO_ADJ = 1e6  # LO offset bandwidth

    def __init__(self, external_clock=True) -> None:
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_rx_antenna("RX2")
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]
        self.channel = 0
        self.txstreamer = self.usrp.get_tx_stream(stream_args)
        self.currate = None
        self.external_clock = external_clock
        if self.external_clock:
            self.usrp.set_time_source('external', 0)  # Time-align multiple USRPs
            self.usrp.set_clock_source('external', 0)  # Frequency-align multiple USRPs

    def tune(self, freq, gain, rate, use_lo_offset=False, gpiomode=None):
        # Set GPIO pins if provided
        if gpiomode is not None:
            self.usrp.set_gpio_attr("FP0", "CTRL", 0)
            self.usrp.set_gpio_attr("FP0", "DDR", 0x10)
            self.usrp.set_gpio_attr("FP0", "OUT", gpiomode)

        # Set rate (if provided)
        if rate:
            self.currate = rate

            # self.usrp.set_master_clock_rate(rate*2)
            print("Setting tx/rx rate = %f" % rate)
            self.usrp.set_tx_rate(rate, self.channel)
            self.usrp.set_rx_rate(rate, self.channel)

        # Set the USRP freq
        if use_lo_offset:
            # Push the LO offset outside of sampling freq range
            lo_off = self.currate/2 + self.LO_ADJ
            self.usrp.set_rx_freq(uhd.types.TuneRequest(freq, lo_off), self.channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(freq, lo_off), self.channel)
        else:
            self.usrp.set_rx_freq(uhd.types.TuneRequest(freq), self.channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(freq), self.channel)

        # Set the USRP gain
        self.usrp.set_rx_gain(gain, self.channel)
        self.usrp.set_tx_gain(gain, self.channel)

    def send_samples(self, samples, rate=None, power_dbm=None):
        # Set the sampling rate if necessary
        if rate and rate != self.currate:
            self.usrp.set_tx_rate(rate, self.channel)

        if power_dbm is not None:
            self.usrp.set_tx_power_reference(power_dbm, self.channel)  #change the transmission power level
        # Metadata for the TX command
        meta = uhd.types.TXMetadata()
        meta.has_time_spec = False
        meta.start_of_burst = True

        # Metadata from "async" status call
        as_meta = uhd.types.TXAsyncMetadata()

        # Figure out the size of the received buffer and make it
        max_tx_samps = self.txstreamer.get_max_num_samps()
        tot_samps = samples.size

        tx_buffer = np.zeros((1, max_tx_samps), dtype=np.complex64)

        tx_samps = 0
        while tx_samps < tot_samps:
            nsamps = min(tot_samps - tx_samps, max_tx_samps)
            tx_buffer[:, 0:0 + nsamps] = samples[:, tx_samps:tx_samps + nsamps]
            if nsamps < max_tx_samps:
                tx_buffer[:, nsamps:] = 0. + 0.j
                meta.end_of_burst = True
            tx_samps += self.txstreamer.send(tx_buffer, meta)
        if self.txstreamer.recv_async_msg(as_meta, self.ASYNC_WAIT):
            if as_meta.event_code != uhd.types.TXMetadataEventCode.burst_ack:
                self.logger.debug("Async error code: %s", as_meta.event_code)
        else:
            self.logger.info("Timed out waiting for TX async.")

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--frequency", help="Frequency to receive samples", default=3534e6, type=float)
    parser.add_argument("-g", "--gain", help="Rx gain in dB", default=70, type=float)
    parser.add_argument("-r", "--rate", help="Rx sample rate", default=220e3, type=float)
    parser.add_argument("-n", "--nsamps", help="Number of samples to receive", default=131072, type=int)
    parser.add_argument("-w", "--sample_wait", help="Time between samples", default=1, type=float)
    parser.add_argument("-o", "--output_dir", help="Data directory", default='data', type=str)
    parser.add_argument("-l", "--use_lo_offset", help="Use low offset", default=True, type=bool)

    parser.add_argument("-nsamps", help="Number of samples", type=int, default=131072)
    parser.add_argument("-wampl", help="Amplitude of the sinusoidal signal", type=float, default=1.0)
    parser.add_argument("-wfreq", help="Frequency of the sinusoidal signal", type=int, default=2e3)
    parser.add_argument("-srate", help="Sampling rate in frequency", type=int, default=220e3)
    parser.add_argument("-bw", help="bandwidth", type=int, default=27.5e3)

    parser.add_argument("-bs", help="number of base station", type=int, default=2)
    parser.add_argument("-p_max_dbm", help="maximum power in dbm", type=int, default=15)

    parser.add_argument("-port", help="socket port number", type=int, default=5050)
    parser.add_argument("-server", help="server ip address", type=str, default="127.0.0.1")

def main():
    args = parse_args()

    PORT = args.port
    SERVER = args.server
    ADDR = (SERVER, PORT)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    send_msg(client, ['MSG_CLIENT_TO_SERVER', 'BS'])

    nsamps = args.nsamps
    wampl = args.wampl
    wfreq = args.wfreq
    srate = args.srate
    bw = args.bw

    org_signal = mk_sine(nsamps, wampl, wfreq, bw)

    radio = Radio()
    # tune radio
    radio.tune(args.frequency, args.gain, args.rate, use_lo_offset=args.use_lo_offset)

    index = 0
    power = args.p_max_dbm
    while True:
        start_time = time.time
        duration = 200  # ms
        while time.time() - start_time < duration:
            radio.send_samples(samples=org_signal, rate=args.rate, power_dbm=power)

            recv_data = recv_msg(client)
            if recv_data:
                power = recv_data[1]
                break
            else:
                pass


if __name__ == "__main__":
    main()


