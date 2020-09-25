import scipy
# import librosa
# import librosa.filters
from utils.libs import hz_to_mel,mel_to_hz,normalize,stft,istft
import numpy as np
from scipy.io import wavfile
from hparams import hparams as hps
import scipy.signal
def fft_frequencies(sr=22050, n_fft=2048):
    
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)
def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)
def mel(sr,n_fft,n_mels=128,fmin=0.0,fmax=None,htk=False,norm="slaney",dtype=np.float32,):
	if fmax is None:
		fmax =float(sr)
		fmax=fmax/2

    # Initialize the weights
    
	n_mels = int(n_mels)
    
	weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    
	fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    
	mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    
	fdiff = np.diff(mel_f)
    
	ramps = np.subtract.outer(mel_f, fftfreqs)

    
	for i in range(n_mels):
        # lower and upper slopes for all bins
        
		lower = -ramps[i] / fdiff[i]
        
		upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        
		weights[i] = np.maximum(0, np.minimum(lower, upper))

    
	if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        
		enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        
		weights *= enorm[:, np.newaxis]
    
	else:
        
		weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    
	if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
		print ("warning")
	return weights


def load_wav(path):
	sr, wav = wavfile.read(path)
	wav = wav.astype(np.float32)
	wav = wav/np.max(np.abs(wav))
	try:
		assert sr == hps.sample_rate
	except:
		print('Error:', path, 'has wrong sample rate.')
	return wav


def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def preemphasis(x):
	return scipy.signal.lfilter([1, -hps.preemphasis], [1], x)


def inv_preemphasis(x):
	return scipy.signal.lfilter([1], [1, -hps.preemphasis], x)


def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - hps.ref_level_db
	return _normalize(S)


def inv_spectrogram(spectrogram):
	'''Converts spectrogram to waveform using librosa'''
	S = _db_to_amp(_denormalize(spectrogram) + hps.ref_level_db)	# Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** hps.power))			# Reconstruct phase

def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hps.ref_level_db
	return _normalize(S)


def inv_melspectrogram(spectrogram):
	mel = _db_to_amp(_denormalize(spectrogram) + hps.ref_level_db)
	S = _mel_to_linear(mel)
	return inv_preemphasis(_griffin_lim(S ** hps.power))


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
	window_length = int(hps.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshold = _db_to_amp(threshold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x:x+window_length]) < threshold:
			return x + hop_length
	return len(wav)


def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(hps.gl_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y


def _stft(y):
	n_fft, hop_length, win_length = _stft_parameters()
	return stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
	_, hop_length, win_length = _stft_parameters()
	return istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
	n_fft = (hps.num_freq - 1) * 2
	hop_length = int(hps.frame_shift_ms / 1000 * hps.sample_rate)
	win_length = int(hps.frame_length_ms / 1000 * hps.sample_rate)
	return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectrogram)
	

def _mel_to_linear(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	inv_mel_basis = np.linalg.pinv(_mel_basis)
	inverse = np.dot(inv_mel_basis, spectrogram)
	inverse = np.maximum(1e-10, inverse)
	return inverse


def _build_mel_basis():
	n_fft = (hps.num_freq - 1) * 2
	return mel(hps.sample_rate, n_fft, n_mels=hps.num_mels)

def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
	return np.power(10.0, x * 0.05)

def _normalize(S):
	return np.clip((S - hps.min_level_db) / -hps.min_level_db, 0, 1)

def _denormalize(S):
	return (np.clip(S, 0, 1) * -hps.min_level_db) + hps.min_level_db

