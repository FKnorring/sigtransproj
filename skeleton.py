"""
Skeleton for the wireless communication system project in Signals and
Transforms

For plain text inputs, run:
$ python3 skeleton.py "Hello World!"

For binary inputs, run:
$ python3 skeleton.py -b 010010000110100100100001
flags: 
    -rate sample frequency of tests
    -t number of tests

2020-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import sys
import re
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import wcslib as wcs

CORRECT_LIMIT = 0.8

def run(data, sample_rate=10000, string_data=True):
    # Parameters
    channel_id = 1      # Channel id
    Tb = 0.03           # Symbol width in seconds
    fs = sample_rate    # Sampling frequency in Hz
    fn = (fs/2)         # Nyquist frequency
    fc = 1000           # Carrier frequency in Hz
    wc = 2*np.pi*fc     # Carrier frequency in rad/s
    Ac = np.sqrt(2)     # Carrier amplitude
    Ts = 1/fs           # Sampling period in seconds
    
    # Convert string to bit sequence or string bit sequence to numeric bit
    # sequence
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([bit for bit in map(int, data)])

    
    # Encode baseband signal
    xb = wcs.encode_baseband_signal(bs, Tb, fs)
    
    # Modulation
    t = np.arange(0, Tb * (len(bs)+3), Ts)
    
    xt = wcs.modulate_signal(xb,wc,Ac,t)

    # Band limiter
    wstop = [850/fn, 1150/fn]
    wpass = [900/fn, 1100/fn]
    gpass = 1
    gstop = 30
    
    numerator, denominator = signal.iirdesign(wp = wpass, ws = wstop, gpass= gpass, gstop = gstop, analog=False, ftype='cheby2')
    xf = signal.lfilter(numerator, denominator, xt)

    # Channel simulation
    yr = wcs.simulate_channel(xf, fs, channel_id)

    # Band limiter reciever
    yf = signal.lfilter(numerator, denominator, yr)
    
    # Low pass filter reciever
    wp_lp = 100/fn
    ws_lp = 150/fn
    
    numerator, denominator = signal.iirdesign(wp = wp_lp, ws = ws_lp, gpass= gpass, gstop = gstop, analog= False, ftype='cheby2')
    
    # IQ demodulation
    yI, yQ = wcs.demodulate_signal(yf, wc, t)
    yb = signal.lfilter(numerator, denominator, yI) + 1j * signal.lfilter(numerator, denominator, yQ)

    ybm = np.abs(yb)
    ybp = np.angle(yb)

    # Baseband and string decoding
    br = wcs.decode_baseband_signal(ybm, ybp, Tb, fs)
    data_rx = wcs.decode_string(br)
    return np.array([data,data_rx,bs,br], dtype=object)

def test_string(s, no_tests, sample_rates):
    amount_right = []
    bit_error = []
    bit_transmitted = []
    bit_error_rate = []

    for i, sample_rate in enumerate(sample_rates):
        amount_right.append(0)
        bit_error.append(0)
        bit_transmitted.append(0)
        bit_error_rate.append(0)
        for _ in range(no_tests):
            sent_str, recv_str, sent_bits, recv_bits = run(s, sample_rate)
            while len(recv_bits) != len(sent_bits): sent_str, recv_str, sent_bits, recv_bits = run(s, sample_rate)
            bit_transmitted[i] += len(sent_bits)
            correct_chars = 0
            for j in range(min(len(recv_str), len(sent_str))):
                if recv_str[j] == sent_str[j]:
                    correct_chars += 1
            
            if len(sent_str) * CORRECT_LIMIT <= correct_chars:
                amount_right[i] +=1
            
            for x in sent_bits:
                try:
                    if sent_bits[x] != recv_bits[x]:
                        bit_error[i] +=1
                except IndexError:
                    bit_error[i] +=1
            
        bit_error_rate[i] = (bit_error[i]/bit_transmitted[i])*100

    return amount_right, bit_error_rate

def print_results(no_tests, amount_right, bit_error_rate, sample_rates):
    for i, sample_rate in enumerate(sample_rates):
        print(f"Sample rate = {sample_rate} Hz")
        print("Number of tests: ", no_tests)
        print("Amount right: ", amount_right[i])
        print("Amount right in %: ", amount_right[i]/no_tests*100, "%")
        print("Bit error rate: ", bit_error_rate[i], "%")
        print("")

def test_suite():
    s_string = "a"
    m_string = "abcde"
    l_string = m_string * 5
    xl_string = l_string * 5

    no_tests = 1000
    sample_rates = [10000, 20000]

    s_amount_right, s_bit_error_rate = test_string(s_string, no_tests, sample_rates)
    m_amount_right, m_bit_error_rate = test_string(m_string, no_tests, sample_rates)
    l_amount_right, l_bit_error_rate = test_string(l_string, no_tests, sample_rates)
    xl_amount_right, xl_bit_error_rate = test_string(xl_string, no_tests, sample_rates)

    print(f"Short string: {s_string}")
    print_results(no_tests, s_amount_right, s_bit_error_rate, sample_rates)
    print(f"Medium string: {m_string}")
    print_results(no_tests, m_amount_right, m_bit_error_rate, sample_rates)
    print(f"Long string: {l_string}")
    print_results(no_tests, l_amount_right, l_bit_error_rate, sample_rates)
    print(f"Extra long string: {xl_string}")
    print_results(no_tests, xl_amount_right, xl_bit_error_rate, sample_rates)
    
def check_args(args):
    bs, sample_rate, tests = "", 10000, 1
    for i, arg in enumerate(args):
        if arg == "-b":
            bs = args[i+1]
            # Check if string only contains 0s and 1s
            regex = re.compile('^[01]+$')
            assert regex.match(bs) is not None, "Invalid string"
        if arg == "-rate":
            sample_rate = args[i+1]
            assert sample_rate.isdigit() is True, "Invalid sample rate"
            sample_rate = int(sample_rate)
        if arg == "-t":
            tests = args[i+1]
            assert tests.isdigit() is True, "Invalid test amount"
            tests = int(tests)
    return bs, sample_rate, tests

def main():
    _, *args = sys.argv
    if len(args) == 0:
        test_suite()
        return
    elif len(args) >= 2 and "-b" in args:
        bs, sample_rate, tests = check_args(args)
        print("Transmitted string: ", wcs.decode_string([int(bit) for bit in bs]))
        print("Sample rate: ", sample_rate)
        for _ in range(tests):
            sbits, rbits = [False], []
            while len(sbits) != len(rbits): _, recv, sbits, rbits = run(bs, sample_rate, False)
            print(f"Received string: {recv}")
        return
    else:
        print("Invalid arguments, \nusage: python3 skeleton.py -b <bits> -rate (optional) <sample rate> -t (optional) <tests>")
        return
    #test_suite() # Run tests

if __name__ == "__main__":
    main()

