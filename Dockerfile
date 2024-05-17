FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG CUDA=11.7
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1 
ARG ONNXRUNTIME_VERSION=1.8.1
ARG MMCV_VERSION=1.6.2
ARG PPLCV_VERSION=0.7.0

ENV DEBIAN_FRONTEND=noninteractive

### change the system source for installing libs
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

### update apt and install libs
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN chmod 777 /tmp
RUN apt-get clean && apt-get update && \
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev curl --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/Miniconda3-latest-Linux-x86_64.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### pytorch
# RUN /opt/conda/bin/conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
# RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
ENV PATH /opt/conda/bin:$PATH

### install mmcv-full
# RUN /opt/conda/bin/pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html  -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /root/workspace
### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

### cp trt from pip to conda
# RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/

### install mmdeploy
# ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
# ENV TENSORRT_DIR=/workspace/tensorrt
# ARG VERSION
# RUN git clone https://github.com/HuangJunJie2017/mmdeploy.git
# RUN cd mmdeploy &&\
#     if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
#     git submodule update --init --recursive &&\
#     mkdir -p build &&\
#     cd build &&\
#     cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
#     make -j$(nproc) &&\
#     cd .. &&\
#     pip install -e .  -i https://pypi.tuna.tsinghua.edu.cn/simple

### build sdk
# RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
#     cd ppl.cv &&\
#     git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
#     ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

# RUN cd /root/workspace/mmdeploy &&\
#     rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc &&\
#     mkdir -p build && cd build &&\
#     cmake .. \
#         -DMMDEPLOY_BUILD_SDK=ON \
#         -DMMDEPLOY_BUILD_EXAMPLES=ON \
#         -DCMAKE_CXX_COMPILER=g++ \
#         -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
#         -DTENSORRT_DIR=${TENSORRT_DIR} \
#         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
#         -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
#         -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
#         -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
#         -DMMDEPLOY_CODEBASES=all &&\
#     make -j$(nproc) && make install &&\
#     export SPDLOG_LEVEL=warn &&\
#     if [ -z ${VERSION} ] ; then echo "Built MMDeploy master for GPU devices successfully!" ; else echo "Built MMDeploy version v${VERSION} for GPU devices successfully!" ; fi

# ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# RUN pip install mmdet==2.25.1 mmsegmentation==0.25.0  -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    # cd ..

RUN pip install numba==0.53.0 \
    numpy==1.23.5 \
    nuscenes-devkit \
