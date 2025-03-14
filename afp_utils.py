def v8_extract_long_fingerprint(song_path, full_duration, chunk_dur=10,sr=32000, overlap=True):
    # A newer version of fpcalc that do the chunks and overlap (vs. manually in v7, below)
    # from https://acoustid.org/chromaprint
    # https://github.com/beetbox/pyacoustid/blob/master/fpcalc.py
    # Usage: fpcalc.exe [OPTIONS] FILE [FILE...]
    # Generate fingerprints from audio files/streams.
    # Options:
    #   -format NAME   Set the input format name
    #   -rate NUM      Set the sample rate of the input audio
    #   -channels NUM  Set the number of channels in the input audio
    #   -length SECS   Restrict the duration of the processed input audio (default 120)
    #   -chunk SECS    Split the input audio into chunks of this duration
    #   -algorithm NUM Set the algorithm method (default 2)
    #   -overlap       Overlap the chunks slightly to make sure audio on the edges is fingerprinted
    #   -ts            Output UNIX timestamps for chunked results, useful when fingerprinting real-time audio stream
    #   -raw           Output fingerprints in the uncompressed format
    #   -signed        Change the uncompressed format from unsigned integers to signed (for pg_acoustid compatibility)
    #   -json          Print the output in JSON format
    #   -text          Print the output in text format
    #   -plain         Print the just the fingerprint in text format
    #   -version       Print version information
    # Note that fpcalc needs ffmpeg in Path. download from https://www.ffmpeg.org/download.html#build-windows

    filename = song_path.replace('\\', '/')
    fpexe = 'PATH TO/fpcalc.exe'
    command = f'"{fpexe}" -raw -chunk {chunk_dur} -overlap -rate {sr} -length {full_duration} "{filename}"'
    if not overlap:
        command = f'"{fpexe}" -raw -chunk {chunk_dur} -rate {sr} -length {full_duration} "{filename}"'

    try:
        fpcalc_output = subprocess.check_output(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running fpcalc: {e}")
        return None

    lines = fpcalc_output.decode("utf-8").splitlines()
    fingerprints = []
    chunk_start=0
    for i in range(0, len(lines), 3):
        # print(f'{lines[i]} {lines[i+1]}')
        if "DURATION=" in lines[i]:
            fpcalc_dur=lines[i].split('=')[1]
            overlap=int(fpcalc_dur)-chunk_dur
            chunk_end=chunk_start+int(fpcalc_dur)
            # print(f'chunk_start:{chunk_start} chunk_end:{chunk_end}  added dur:{dur}')
        if "FINGERPRINT=" in lines[i + 1]:
            fingerprint = [int(x) for x in lines[i + 1].split('=')[1].split(',')]

        fingerprints.append({
            "fingerprint": fingerprint,
            "time_offset": f"{chunk_start},{chunk_end}"
        })
        chunk_start = chunk_end-overlap
    return fingerprints
# ---------------------------------------------------------------------------------------------
def compute_log_bins(magnitude, num_bins=32):
    bins = np.logspace(np.log10(1), np.log10(len(magnitude)), num_bins)
    binned_energy = [np.sum(magnitude[int(bins[i]):int(bins[i+1])]) for i in range(len(bins) - 1)]
    return binned_energy
# ---------------------------------------------------------------------------------------------
def highpass_filter(audio, sr, cutoff=200):
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    if normalized_cutoff >= 1:
        # If the cutoff is too high, return the original audio
        # print(f"Warning: Cutoff frequency {cutoff} Hz is too high for sample rate {sr} Hz. Skipping filter.")
        return audio
    b, a = butter(1, normalized_cutoff, btype='high')
    return lfilter(b, a, audio)
# ---------------------------------------------------------------------------------------------
def extract_robust_fingerprints(audio, sr, window_size=1024, hop_size=512):
    # Resampling on 8192Hz and 1024 window size are recommended here:
    # https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/projects/2019/AudioFingerprinting.pdf
    # Window size 1024:
    #     The FFT algorithm is most efficient when the window size is a power of two (e.g., 256, 512, 1024, 2048, 4096).
    # Duration of window:
    #     At Sample rate = 8192 Hz (Samples per second, Window size=1024/8192. is about 0.125 seconds
    # Hop Size:
    #     is Half of window size, to create overlapping and match samples at any location and not miss edges
    #     Hop duration (half of window size) = 0.0625 seconds
    # Number of Windows:
    #     Assume duration of song is 1 hour (3600 seconds)
    #     So there are 3600/0.125 = 28800 windows, or 57600 fingerprints (vs 77,502 on v_6.0 at 4096 window size and 44100Hz

    fingerprints = []
    for start in range(0, len(audio) - window_size, hop_size):
        window = audio[start:start + window_size]
        # 1: Apply high-pass filter
        window = highpass_filter(window, sr)
        # 2: Compute FFT and magnitude
        fft_result = np.fft.rfft(window)
        magnitude = np.abs(fft_result)
        # 3: Frequency binning
        binned_energy = compute_log_bins(magnitude)
        # 4: Identify top peaks
        peaks = np.argpartition(binned_energy, -5)[-5:]
        peaks_sorted = sorted(peaks, key=lambda x: binned_energy[x], reverse=True)
        # 5: Hash peaks
        fingerprint = hash(tuple(peaks_sorted))
        fingerprints.append({"hash": fingerprint, "time_offset": start / sr})
    return fingerprints
# ---------------------------------------------------------------------------------------------
