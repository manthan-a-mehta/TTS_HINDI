import numpy as np
from utils.libs2 import get_window,valid_audio,pad_center,frame,dtype_r2c,fix_length,dtype_c2r,window_sumsquare
class LibrosaError(Exception):
    """The root librosa exception class"""

    pass

class ParameterError(LibrosaError):
    """Exception class for mal-formed inputs"""

    pass
def hz_to_mel(frequencies, htk=False):
    

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels
def mel_to_hz(mels, htk=False):
    

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
   
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm
def tiny(x):

    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
def stft(
    y,
    n_fft=4000,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            print("not ok")

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix

__FFTLIB = None


def set_fftlib(lib=None):
    

    global __FFTLIB
    if lib is None:
        from numpy import fft

        lib = fft

    __FFTLIB = lib


def get_fftlib():
    """Get the FFT library currently used by librosa
    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    global __FFTLIB
    set_fftlib(None)
    return __FFTLIB
def istft(
    stft_matrix,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    length=None,
):

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if dtype is None:
        dtype = dtype_c2r(stft_matrix.dtype)

    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    fft = get_fftlib()

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window,
        n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[start:], length)

    return y
def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample : (sample + n_fft)] += ytmp[:, frame]
