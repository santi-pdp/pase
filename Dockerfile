FROM ubuntu:18.04
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git make cmake python3-pip
RUN git clone https://github.com/drowe67/codec2.git
RUN cd codec2 && \
    mkdir build_linux && \
    cd build_linux && \
    cmake .. && \
    make && \
    make install
RUN pip3 install cython numpy
RUN git clone https://github.com/edwarddixon/pycodec2 && \
    cd pycodec2 && \
    python3 setup.py install
