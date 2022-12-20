#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton for the wireless communication system project in Signals and
Transforms

For plain text inputs, run:
$ python3 skeleton.py "Hello World!"

For binary inputs, run:
$ python3 skeleton.py -b 010010000110100100100001

:)
"""

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import l as wcs
import time

# https://www.geeksforgeeks.org/design-iir-bandpass-elliptic-filter-using-scipy-python/
# Function to depict magnitude
# and phase plot
def plotFilter(b, a, Fs):

    # Compute frequency response of the
    # filter using signal.freqz function
    wz, hz = signal.freqz(b, a)

    # Calculate Magnitude from hz in dB
    Mag = 20*np.log10(abs(hz))

    # Calculate phase angle in degree from hz
    Phase = np.unwrap(np.arctan2(np.imag(hz),
                                np.real(hz)))*(180/np.pi)

    # Calculate frequency in Hz from wz
    Freq = wz*Fs/(2*np.pi)

    # Plot filter magnitude and phase responses using subplot.
    fig = plt.figure(figsize=(10, 6))

    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.set_title('Magnitude Response')
    sub1.set_xlabel('Frequency (Hz)')
    sub1.set_ylabel('Magnitude (dB)')
    plt.xlim(2000*ANGULAR,3000*ANGULAR)

    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)')
    sub2.set_xlabel(r'Frequency (Hz)')
    sub2.set_title(r'Phase response')
    sub2.grid()

    plt.xlim(2000*ANGULAR,3000*ANGULAR)
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()

def testSuite(FS, b, a):
    tnew = np.arange(0,1,1/FS) # freqs = samp freq
    xnew = np.sin(2500*ANGULAR*tnew)
    xt = signal.lfilter(b,a,xnew)

    plot(tnew, xnew, r"Test")
    plot(tnew, xt, r"filtered Test")

def plot(x, y, label, yLabel, xLabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, label=f"{label}")
    ax.grid(color="lightgrey", linestyle="dotted")
    ax.legend(loc="upper right")
    ax.set_ylabel(f"{yLabel}")
    ax.set_xlabel(f"{xLabel}")
    ax.set_xlim(-0.001, 0.331)

def plotStem(x, y ,label):
    fig, ax = plt.subplots()
    markerline, stems, baseline = ax.stem(x, y, label=f"{label}")
    #ax.set_xlim(0, 0.0012)
    for stem in stems:
        stem.set_linewidth(0.5)
    plt.setp(markerline, markersize = 0.5)
    ax.legend("upper right")

# -Constants--------------------------------------------------------------------------
ANGULAR = 2*np.pi               # Hz or rad/s 
TB = .03                        # Period of baseband signal
TC = TB/75                      # .0004
FS = 12500                      # Sampling frequency in Hz 12500
TS = 1/FS                       # Sampling time
WS = ANGULAR/TS                 # Sampling frequency in rad/s
WN = (WS/2)                     # Nyquist frequency
WC = 2500*ANGULAR               # Cutoff frequency (2500 Hz) 
KB = int(TB*FS)                 # Symbol width in samples

def system():
    # Detect input or set defaults
    string_data = True
    if len(sys.argv) == 2:
        data = str(sys.argv[1])
    elif len(sys.argv) == 3 and str(sys.argv[1]) == '-b':
        string_data = False
        data = str(sys.argv[2])
    else:
        #print('Warning: No input arguments, using defaults.', file=sys.stderr)
        data = "Shalooooooooom, för enkelt!!!"

    # Convert string to bit sequence or string bit sequence to numeric bit
    # sequence
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([bit for bit in map(int, data)])

    # -Encoded signal (xb)-----------------------------------------------------------------
    xb = wcs.encode_baseband_signal(bs, KB)
    nB = (len(bs)+3) # Number of bits in the encoded message
    TIME = np.arange(0, nB*TB, TS) 
    
    plot(TIME, xb, "$x_b$", "$x_b(t)$", "$t / s")
  
    # -Carrier (xc)-------------------------------------------------------------------------
    Ac = 1
    xc = Ac*np.sin(WC*TIME)
    #plot(TIME, xc, "$x_c$")

    # -Modulater (xm)-------------------------------------------------------------------------
    xm = xb*xc
    #plot(TIME, xm, "$x_m$")

    # -Band-pass filter specifications------------------------------------------------------------------------
    gpass = 2 # dB
    wpass1 = 2400*ANGULAR   # 2400
    wpass2 = 2600*ANGULAR   # 2600
    gstop = 40 # dB
    wstop1 = 2300*ANGULAR   # 2300
    wstop2 = 2700*ANGULAR   # 2700
    
    wpass = [wpass1/WN, wpass2/WN]
    wstop = [wstop1/WN, wstop2/WN]

    N, wn = signal.ellipord(wpass, wstop, gpass, gstop)
    b, a = signal.ellip(N, gpass, gstop, wn, btype='bandpass') #,output='zpk'
    
    #plotFilter(b,a,ws)
    #testSuite(FS, b, a)

    # -Filtered modulated signal (xt)--------------------------------------------
    xt = signal.lfilter(b, a, xm)

    plot(TIME, xt, "Filtered modulated signal $x_t$", "$x_t(t)$","$t / s$")

    # -Channel simulation (yr)--------------------------------------------------------
    yr = wcs.simulate_channel(xt, FS)

    plot(TIME, yr, "$simulation y_r$", "$y_r(t)$", "$t / s$") 

    # -Filtered recieved signal (ym)-------------------------------------------------
    ym = signal.lfilter(b, a, yr)
    plot(TIME, ym, "filtered recieved signal $y_m$","$y_m(t)$","$t / s$")

    # -Low-pass filter specifications
    gpass = 2 # dB
    wpass = (100*ANGULAR)/WN
    gstop = 40 # dB
    wstop = (150*ANGULAR)/WN

    N, wn = signal.ellipord(wpass, wstop, gpass, gstop)
    b, a = signal.ellip(N, gpass, gstop, wn, btype='lowpass') #,output='zpk'

    # -Demodulator-------------------------------------------------------------------------------
    yI = ym*np.cos(WC*TIME)
    yQ = -ym*np.sin(WC*TIME)

    # Low-pass filter the I and Q parts
    yI = signal.lfilter(b, a, yI)
    yQ = signal.lfilter(b, a, yQ)

    yb = yI + 1j*yQ

    ybPhase = np.angle(yb) #Symbol information
    ybMag = np.abs(yb) #Detect transmissions

    plot(TIME, ybPhase, "$\\angle y_b$", "$\\angle y_b(t)$", "$t / s$")
    #plot(TIME, ybMag, "$|y_b|$")

    # -Decoder------------------------------------------------------------------------------------
    br = wcs.decode_baseband_signal(ybMag, ybPhase, KB)
    data_rx = wcs.decode_string(br)

    plt.show()
    return np.array([data, data_rx, bs, br], dtype=object)

def main():
    times = 1000
    numErrorsExact = 0
    numErrorsBit = 0
    numBits = 0
    unshiftedStrBits = 0
    totUnshiftedStrBits = 0

    for i in range(0, times):
        recieved = system()
        err = recieved[0] != recieved[1]
        print(f'{recieved[0]} ({i})')
        print(recieved[1])
        if err:
           numErrorsExact+=1 

        tempErr = 0 
        for i in recieved[2]:
            numBits+=1
            err = recieved[2][i] != recieved[3][i]
            if err:
                tempErr+=1
                numErrorsBit+=1
        
        if tempErr/len(recieved[2]) < .5:
            unshiftedStrBits += tempErr
            totUnshiftedStrBits += len(recieved[2])

    procentUnshifted = unshiftedStrBits/totUnshiftedStrBits        
    procent = numErrorsExact/times
    procentBits = numErrorsBit/numBits

    print(f'Error rate exact: {procent*100}% of {times} samples')
    print(f'Error rate bits:  {procentBits*100}% of {numBits} bits')
    #There is a probability that more than half of the bits in a string is wrong but 
    #that is a low probability
    print(f'Error rate unshifted bits:  {procentUnshifted*100}% of {times} samples') 
    


if __name__ == "__main__":
    main()