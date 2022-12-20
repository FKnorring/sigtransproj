#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the wireless communication system project in Signals and Transforms

2020-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import numpy as np
from scipy import signal
from scipy.stats import chi2

# List of channels and their max average power [fl, fu, Pmax]^T
_channels = np.array([
    [np.nan,  900, 1150, 1300, 1550, 1725, 1950, 2100, 2400, 2700, 3050, 3200, 3475, 3550, 3750, 3900, 4150, 4300, 4550, 4750, 4900],
    [np.nan, 1100, 1250, 1500, 1650, 1875, 2050, 2300, 2600, 2900, 3150, 3400, 3525, 3650, 3850, 4100, 4250, 4500, 4650, 4850, 5100],
    [np.nan,   30,   33,   27,   33,   33,   33,   30,   27,   30,   33,   33,   27,   33,   30,   30,   33,   27,   33,   30,   27]
])

def encode_string(instr):
    """
    Converts a string to a binary numpy array.

    Parameters
    ----------
    instr : str
        A UTf-8 encoded Python string.

    Returns
    -------
    binary : numpy.array
        A binary array encoding the string.
    """
    tmp = [np.uint8(ord(c)) for c in instr]
    return np.unpackbits(tmp)


def decode_string(inbin):
    """
    Converts a binary numpy array to string.

    Parameters
    ----------
    inbin : numpy.array
        A binary array of ones and zeros encoding a string.
    
    Returns
    -------
    outstr : str
        A UTF-8 encoded Python string.
    """
    tmp = np.packbits(inbin)
    outstr = "".join([chr(b) for b in tmp])
    return outstr


# Encode a bit sequence into a baseband signal
def encode_baseband_signal(b, Tb, fs):
    """
    Encodes a binary sequence into a baseband signal. In particular, generates 
    a discrete-time signal that encodes the binary signal `b` into pulses of 
    width `Tb` seconds at a sampling frequency `fs`, of amplitude 1 (for bits 
    that are 1) and -1 (for bits that are 0).

    To generate the discrete-time signal, the function determines the pulse 
    width in samples that corresponds to the specified parameters `Tb` and 
    `fs`. This is calculated by

        Kb = np.floor(Ts*fs),

    where `Kb` is the pulse width in samples.

    The function also prepends the bits [1, 0] (encoded as [-1, 1]) to the 
    message. These bits are used as a known sequence of bits and used in the 
    decoder (on the receiving side) to determine the time delay between the 
    sender and receiver to synchronize the decoding process with the signal. 

    Parameters
    ----------
    b : numpy.array
        A binary array of 1s and 0s encoding a message.
    Tb : float
        Pulse width in seconds to encode the bits to.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    xb : numpy.array
        Encoded baseband signal.
    """

    # Prepend synchronization sequence and a trailing zero
    b = np.concatenate(([1, 0], b, [0]))

    # Encode bit values
    s = [-1, 1]
    b[b == 0] = s[0]
    b[b == 1] = s[1]

    # Expand
    Kb = int(np.floor(Tb*fs))
    Nx = b.shape[0]
    xb = np.zeros(Nx*Kb)
    xb[np.arange(0, Nx*Kb, Kb)] = b

    # "Lowpass filtering"
    b = np.ones(Kb)
    xb = signal.lfilter(b, 1, xb)

    return xb


# Decode a baseband signal into a bit sequence
def decode_baseband_signal(xm, xp, Tb, fs):   #Kb, sigma2=0.01, s=[-1, 1]):
    """
    Decodes an IQ-demodulated baseband signal consisting of a magnitude signal
    `xm` and a phase signal `xp` into a binary bit sequence.
    
    The bit sequence is recovered by first determining the start of the 
    transmission. This is achieved by comparing the magnitude signal `xm` to a
    threshold, where the threshold is determined using the tail probability of 
    a chi-squared distribution (in essence, the test checks whether there is a 
    signal or only noise with a probability of 99 %).

    Then, a filter with impulse response consisting of two pulses corresponding
    to the synchronization sequence [1, 0] is used to find the first occurence 
    of this pulse sequence in the signal. This is used to determine the time
    delay introduced by the transmission.

    Finally, the bit value is recovered by looking at the sign of the phase 
    signal every `Tb` seconds, corrected for possible sign flips due to the 
    phase shift introduced by the transmission.

    Note that the decoder could be improved in many way and this is a quite
    basic approach to decoding the signal. However, it is a bit easier to
    understand compared to more complicated approaches.

    Parameters
    ----------
    xm : numpy.array
        The magnitude of the IQ-demodulated baseband signal.
    xp : numpy.array
        The phase of the IQ-demodulated baseband signal.
    Tb : float
        Pulse width in seconds to encode the bits to.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    b : numpy.array
        A binary array of 1s and 0s encoding a message.
    """
    # Signal detection
    x = xm**2/np.var(xm[0:100])
    d = chi2.cdf(x, 2) > 0.99

    # Synchronize using a matched filter. N.B: Expects the first to bits to be
    # [1, 0] as prepended by encode_baseband_signal()
    Kb = int(np.floor(Tb*fs))
    b = np.concatenate((np.ones(Kb), -np.ones(Kb)))
    xd = np.sign(xp)*d
    xs = signal.lfilter(b, 1, xd)
    k0 = np.argmax(abs(xs/(2*Kb)) > 0.75)

    # Get bits and change possible 180 degree phase shifts by normalizing with
    # the -first- second bit's sign (b/c of the synchronization sequence '1 0',
    # we know that the -first- second bit must be a -'1'- '0' and thus the
    # symbol -'1'- '-1'). Also strip the first two bits that are used for
    # synchronization.
#    b = np.sign(xp[k0+int(0.9*Kb)::Kb])
#    b = b[2:]/b[0]
    b = np.sign(xp[k0::Kb])
    b = b[1:]/(-b[0])

    return b > 0


# Channel simulation
def simulate_channel(x, fs, channel_id, SNR=3.0, eta=0.25, dmax=5.0):
    """
    Takes the modulated (discrete-time) signal `x` (generated at sampling 
    frequency `fs`) and simulates a wireless transmission through open space at
    a random distance in the interval from 0 to `dmax` meter. The channel model
    consists of

    * Signal attenuation and delay due to its propagation in space,
    * random noise, and
    * random out-of-channel interference.

    Channel model: The chanel model is based on an exponential decay 
    (attenuation) and time-of-flight-based time-delay, that is,

        y[k] = e^(-eta*d)*x[k - m]

    where `d` is a random distance uniformly drawn on the interval [0, `dmax`]
    and

        m = np.round(d/c*fs)

    Random noise: Random noise is added such that a constant (transmitter) 
    signal-to-noise ratio of `SNR` is achieved.

    Out-of-channel interference: In addition to the random noise, an out-of-
    channel interference is added, which corresponds to a random transmission
    at another frequency. The transmission frequency is sampled uniformly from
    the whole spectrum outside the given channel, while the amplitude is 
    sampled from a Gaussian distribution with mean 1 and standard deviation 
    0.2.

    /!\ The default values for `SNR`, `eta`,  as well as `dmax` should not be
        changed unless you know what you are doing. /!\

    Parameters
    ----------
    x : numpy.array
        The modulated signal to be transmitted.

    fs : float
        Sampling frequency.

    channel_id : int
        The id of the communication channel.

    SNR : float, default 3.0
        The signal-to-noise ratio at the transmitter (in dBm). The default is 
        3 dBm, which corresponds to a signal that is twice as strong (in terms
        of power) as the noise.

    eta : float, default 0.25
        Fading coefficient.

    dmax : float, default 5.0
        The maximum transmission distance.

    Returns
    -------
    y : numpy.array
        The signal received by the receiver.
    """

    # Get channel parameters
    if not (channel_id >= 1 and channel_id < _channels.shape[1]):
        raise ValueError(f'channel_id must be between 1 and {_channels.shape[1]-1}, but {channel_id} given.')
    channel = _channels[:, channel_id]

    # Create the channel impulse response: A Kronecker delta with amplitude 
    # exp(-eta*d) at sample m
    c = 340
    d = dmax*np.random.rand(1)
    m = int(np.round(d/c*fs))
    h = np.zeros(m+1)
    h[m] = np.exp(-eta*d)

    # Calculate the noise variance based on the SNR
    # sigma2 = 0.5 => 27 dBm; at Pmax = 30 dBm => SNR = 3 dBm
    # sigma2 = 10^((Pmax-SNR)/10)*1e-3
    sigma2 = 10**((channel[2] - SNR)/10)*1e-3
    Nx = x.shape[0]
    vn = np.sqrt(sigma2)*np.random.randn(Nx)

    # Add out-of-band interference at a random frequency, uniformly distributed
    # outside the channel's frequency band. First, determine the probability of
    # the interference being below or above the channels frequency band.
    fmin = np.min(_channels[0, 1:])
    fmax = np.max(_channels[1, 1:])
    dfbelow = channel[0]-fmin
    dfabove = fmax-channel[1]
    pbelow = dfbelow/(dfbelow+dfabove)

    # Then, sample the interference frequency...
    if np.random.rand(1) <= pbelow:
        # ...below the channel's frequency band with probability pbelow
        fi = fmin + dfbelow*np.random.rand(1)
    else:
        # ...or above the channel's frequency band with probability 1-pbelow
        fi = channel[1] + dfabove*np.random.rand(1)

    # Now, sample the interference amplitude with a mean of 1 (30 dBm) and 
    # a standard deviation of 0.2 (95 % between 0.6 and 1.4). Then add 
    # everything together to generate the interference signal
    Ai = 1 + 0.2*np.random.rand(1)
    k = np.arange(0, x.shape[0])
    vi = Ai*np.sin(2*np.pi*fi*k/fs)

    # Construct received signal
    y = signal.lfilter(h, 1, x) + vn + vi

    return y

# Modulation 
def modulate_signal(bits,wc,Ac,t):
    #t = np.arange(0, Tb * len(bits), Tb)
    xt = bits * Ac * np.sin(wc * t)
    return xt

# IQ demodulation 
def demodulate_signal(ym,wc,t):
    yi = ym * np.cos(wc * t)
    yq = -ym * np.sin(wc * t)
    return yi,yq