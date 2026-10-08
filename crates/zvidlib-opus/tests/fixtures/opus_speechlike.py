"""Writes a deterministic speech-like mono signal as 48 kHz f32le.

A glottal pulse train with a gliding pitch, shaped by three formant
resonators whose centers move between vowels, with an unvoiced (fricative)
noise burst in the middle. Opus's mode decision treats it as speech, so a
low-rate encode of it codes in SILK or hybrid mode rather than CELT.
"""
import math
import random
import struct
import sys

RATE = 48_000
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
N = int(RATE * SECONDS)
random.seed(1234)

VOWELS = [(700, 1220, 2600), (300, 2300, 3000), (500, 900, 2400), (400, 1900, 2550)]


def resonator(freq, bandwidth):
    r = math.exp(-math.pi * bandwidth / RATE)
    theta = 2 * math.pi * freq / RATE
    return 2 * r * math.cos(theta), -r * r


out = []
state = [[0.0, 0.0] for _ in range(3)]
phase = 0.0
for i in range(N):
    t = i / RATE
    pitch = 110 + 40 * math.sin(2 * math.pi * 1.3 * t)
    phase += pitch / RATE
    voiced = not (0.45 * SECONDS < t < 0.6 * SECONDS)
    if voiced:
        excitation = 1.0 if phase >= 1.0 else 0.0
    else:
        excitation = (random.random() - 0.5) * 0.3
    if phase >= 1.0:
        phase -= 1.0
    vowel = VOWELS[int(t * 4) % len(VOWELS)]
    sample = excitation
    for k, (formant, s) in enumerate(zip(vowel, state)):
        a1, a2 = resonator(formant, 80 + 40 * k)
        y = sample + a1 * s[0] + a2 * s[1]
        s[1], s[0] = s[0], y
        sample = y * 0.12
    envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t) ** 2
    out.append(sample * envelope)

peak = max(abs(v) for v in out)
out = [0.5 * v / peak for v in out]

with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<%df' % len(out), *out))
