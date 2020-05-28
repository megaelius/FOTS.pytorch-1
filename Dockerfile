FROM sc-hw-artf.nvidia.com/sw-nvdl-docker/pytorch:20.05-py3

COPY . /build-context

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver

RUN mkdir -p /workspace

WORKDIR /workspace

# RUN git clone https://github.com/NVIDIA/apex.git
# WORKDIR /workspace/apex
# RUN git checkout 55716d858878ec692d0f373a8dbb39a29f0952a5
# RUN pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# WORKDIR /workspace

RUN apt-get install -y nasm

RUN conda install -y pyyaml

RUN yes | pip install msgpack coolname py3nvml jsonnet

# Install libjpeg-turbo which will provide much more efficient jpeg loading
RUN git clone -b 2.0.3 https://github.com/libjpeg-turbo/libjpeg-turbo.git
RUN mkdir -p /workspace/libjpeg-turbo/build
WORKDIR /workspace/libjpeg-turbo/build
RUN cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
RUN make install -j
# This gives us access to libjpeg-turbo from python
# https://pypi.org/project/PyTurboJPEG/
RUN yes | pip install PyTurboJPEG

WORKDIR /workspace

RUN git clone -b v0.14.0 https://github.com/google/jsonnet.git
WORKDIR /workspace/jsonnet
RUN make install -j

WORKDIR /workspace

RUN pip install Shapely

RUN yes | pip install editdistance

RUN yes | pip install cycler pygame pytz wget bson libscrc ffmpeg-python Augmentor

RUN conda install -y opencv

ARG ADLR_OPS_VER=unknown2
COPY deploy_key .
ADD https://gitlab-master.nvidia.com/api/v4/projects/19842/repository/branches/master?private_token=X4x4WTSGu32rc-JHgAQQ /tmp/devalidateCache
RUN chmod 600 deploy_key && \
    eval $(ssh-agent) && \
    ssh-add deploy_key && \
    mkdir -p /root/.ssh && \
    echo -e "Host 172.20.248.201\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config && \
    ssh-keyscan -H 172.20.248.201 >> /root/.ssh/known_hosts && \
    git clone ssh://git@172.20.248.201:12051/ADLR/pytorch_image_ops.git
WORKDIR /workspace/pytorch_image_ops
RUN bash ccache_install.sh
RUN PATH=/root/ccache/lib:$PATH
RUN CUDA_NVCC_EXECUTABLE=/root/ccache/cuda/nvcc
RUN python setup.py install

WORKDIR /workspace

RUN yes | pip uninstall pillow
# Version 7 breaks compatibility with torchvision<0.5.0
RUN yes | CC="cc -mavx2" pip install -U --force-reinstall "pillow-simd"

RUN apt-get update

WORKDIR /build-context/rroi_align
RUN python setup.py install

WORKDIR /workspace
