"""
FFT class
"""
import numpy as np
import matplotlib.pyplot as plt
from . import feature_module
from . import gen_stft_win

hop = 160
fftsize = 512
winsize = 480
overlap_size = fftsize - hop

class stft_class:
    """
    fft_module
    """
    def __init__(self,
                 hop = 160,
                 fftsize=512,
                 winsize=480):
        self.hop = hop
        self.fftsize = fftsize
        self.winsize = winsize
        self.overlap_size = fftsize - hop
        self.ibuf = np.zeros(winsize)
        self.obuf = np.zeros(winsize)
        self.win = gen_stft_win.gen_stft_win(winsize, hop, fftsize)

    def reset(self):
        """
        reset states
        """
        self.ibuf *= 0
        self.obuf *= 0

    def stft_frame_proc(self, data_time):
        """
        stft_frame_proc
        """
        self.ibuf[self.overlap_size:] = data_time
        data_freq = np.fft.rfft(self.ibuf * self.win, self.fftsize)
        self.ibuf[:self.overlap_size] = self.ibuf[self.hop:]
        return data_freq

    def istft_frame_proc(self, data_freq, tfmask = 1.0):
        """
        istft_frame_proc
        """
        data_freq = data_freq * tfmask
        data = np.fft.irfft(data_freq)[:self.winsize]
        wdata = data * self.win
        self.obuf += wdata
        odata = self.obuf[:self.hop].copy()
        self.obuf[:self.overlap_size] = self.obuf[self.hop:]
        self.obuf[self.overlap_size:] = 0
        return odata

win = gen_stft_win.gen_stft_win(winsize, hop, fftsize)
a = np.random.uniform(-1,1,10000)
A = feature_module.strided_app(a, winsize, hop)
print(a)

print(A)
ff =np.fft.rfft(A * win, fftsize)
print(ff.shape)

iff = np.fft.irfft(ff)[:,:winsize]
tt = iff * win

o = np.zeros(winsize)
out = []
for t in tt:
    o = o + t
    out += [o[:hop].copy()]
    o[:overlap_size] = o[hop:]
    o[overlap_size:] = 0
out = np.array(out)
print(out.shape)
out = out.flatten()[overlap_size:]
aa = a[:out.shape[0]]
print(out.shape)
print(aa.shape)
print(np.max(np.abs(out-aa)))
print(out)

plt.plot(out)
plt.plot(aa)

plt.show()
