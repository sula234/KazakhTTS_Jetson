# Kazakh_TTS on Jetson Orin (JetPack 5.1.2 / L4T 35.4.1 + PyTorch)
FROM dustynv/l4t-pytorch:r35.4.1

SHELL ["/bin/bash","-lc"]
WORKDIR /workspace

# Basic tools, audio utils, HDF5 dev libs (needed for h5py build on aarch64)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs curl jq unzip locales ca-certificates \
    sox ffmpeg nano vim-tiny \
    build-essential pkg-config libhdf5-dev libhdf5-serial-dev hdf5-tools \
 && rm -rf /var/lib/apt/lists/*

# Locales (kk/ru/en)
RUN sed -i 's/# \(kk_KZ.UTF-8 UTF-8\)/\1/; s/# \(ru_RU.UTF-8 UTF-8\)/\1/; s/# \(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ESPnet source + Kazakh_TTS recipe
RUN git clone --depth=1 https://github.com/espnet/espnet.git /workspace/espnet
RUN cd /workspace/espnet/egs2 && git clone --depth=1 https://github.com/IS2AI/Kazakh_TTS.git

# Required symlinks for the recipe
RUN set -eux; cd /workspace/espnet/egs2/Kazakh_TTS/tts1; \
    ln -sf ../../TEMPLATE/tts1/path.sh .; \
    ln -sf ../../TEMPLATE/asr1/pyscripts .; \
    ln -sf ../../TEMPLATE/asr1/scripts .; \
    ln -sf ../../../tools/kaldi/egs/wsj/s5/steps .; \
    ln -sf ../../TEMPLATE/tts1/tts.sh .; \
    ln -sf ../../../tools/kaldi/egs/wsj/s5/utils .

# Use system HDF5 when building h5py on ARM64
ENV HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial

# Python deps:
# - h5py built against system HDF5
# - pin numpy/scipy for PWG
# - add parallel_wavegan and common audio/text libs
RUN pip3 install --no-cache-dir --no-binary=h5py 'h5py==3.8.0' && \
    pip3 install --no-cache-dir "numpy<2" "scipy==1.10.1" \
        soundfile librosa num2words nltk parallel_wavegan

# Make ESPnet importable everywhere (so espnet2 is found)
# You already installed numpy/scipy/nltk/parallel_wavegan/etc earlier…
RUN pip3 install --no-cache-dir --no-deps -e /workspace/espnet

# Extra runtime deps for ESPnet TTS (installed w/o deps to avoid torch replacement)
RUN pip3 install --no-deps \
    typeguard humanfriendly sentencepiece pyyaml  espnet-tts-frontend \
    phonemizer jamo pykakasi jieba unidecode inflect \
    torch_complex g2p_en more_itertools pyworld jaconv

# after cloning ESPnet
ENV PYTHONPATH=/workspace/espnet:$PYTHONPATH


# Helpful message
CMD echo $'Container ready (Jetson).\nNext:\n  cd /workspace/espnet/egs2/Kazakh_TTS/tts1\n  python3 synthesize.py --text "ассаляму алейкум ва рахматуллахи ва баракатуху"'
