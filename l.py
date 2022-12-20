#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the wireless communication system project in Signals and Transforms

2020 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""


import numpy as np
from scipy import signal
from scipy.stats import chi2


# Conversion of string to binary numpy array
def encode_string(instr):
    tmp = [np.uint8(ord(c)) for c in instr]
    return np.unpackbits(tmp)


# Conversion of binary numpy array to string
def decode_string(inbin):
    tmp = np.packbits(inbin)
    outstr = "".join([chr(b) for b in tmp])
    return outstr


# Encode a bit sequence into a baseband signal
def encode_baseband_signal(b, Kb, s=[-1, 1]):
    # Prepend synchronization sequence and a trailing zero
    b = np.concatenate(([1, 0], b, [0]))

    # Encode bit values
    b[b == 0] = s[0]
    b[b == 1] = s[1]

    # Expand
    Nx = b.shape[0]
    xb = np.zeros(Nx*Kb)
    xb[np.arange(0, Nx*Kb, Kb)] = b

    # "Lowpass filtering"
    b = np.ones(Kb)
    xb = signal.lfilter(b, 1, xb)

    return xb


# Decode a baseband signal into a bit sequence
# TODO: Should take 's' into account. Hardcoded to -1/1 now.
def decode_baseband_signal(xm, xp, Kb, sigma2=0.01, s=[-1, 1]):
    # Signal detection
    x = xm**2/np.var(xm[0:100])
    d = chi2.cdf(x, 2) > 0.99

    # Synchronize using a matched filter. N.B: Expects the first to bits to be
    # [1, 0] as prepended by encode_baseband_signal()
    b = np.concatenate((np.ones(Kb), -np.ones(Kb)))
    xd = np.sign(xp)*d
    xs = signal.lfilter(b, 1, xd)
    k0 = np.argmax(abs(xs/(2*Kb)) > 0.75)

    # (Quick&dirty) synchronization (old code)
    # x = xm**2/np.var(xm[0:100])
    # xb = chi2.cdf(x, 2) > 0.99
    # k0 = np.argmax(xb)

    # Get bits and change possible 180 degree phase shifts by normalizing with
    # the -first- second bit's sign (b/c of the synchronization sequence '1 0',
    # we know that the -first- second bit must be a -'1'- '0' and thus the
    # symbol -'1'- '-1'). Also strip the first two bits that are used for
    # synchronization.
    # b = np.sign(xp[k0+int(0.9*Kb)::Kb])
    # b = b[2:]/b[0]
    b = np.sign(xp[k0::Kb])
    b = b[1:]/(-b[0])

    return b > 0


# Channel simulation
def simulate_channel(x, fs, channel_id=0, sigma2=0.25, dmax=5):
    # Propagation
    c = 340
    d = dmax*np.random.rand(1)
    Td = d/c
    Kd = int(np.round(Td*fs))
    b = np.zeros(Kd+1)
    b[Kd] = np.exp(-0.25*d)

    # Noise
    Nx = x.shape[0]
    vn = np.sqrt(sigma2)*np.random.randn(Nx)

    # Interference
    # fi = 900+(4250-900)*np.random.rand(3)
    # Ai = 0.25 + 0.5*np.random.rand(3)
    # vi = np.inner(Ai, np.sin(2*np.pi*fs*np.outer(fi, np.arange(0, x.shape[0]))).T)
    vi = 0

    # Construct received signal
    y = signal.lfilter(b, 1, x) + vn + vi

    return y