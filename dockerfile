FROM ubuntu:18.04
ENV http_proxy="http://proxy-us.intel.com:911" \
    https_proxy="http://proxy-us.intel.com:912"
COPY apt.conf /etc/apt/apt.conf
RUN apt-get update && \
    apt-get install -y python && \
    apt-get install -y git make cmake python3-pip sox libsox-dev libsox-fmt-all
RUN git config --global http.proxy http://proxy-us.intel.com:911 && \
    git config --global https.proxy http://proxy-us.intel.com:912 && \
    git clone https://github.com/drowe67/codec2.git
RUN cd codec2 && \
    mkdir build_linux && \
    cd build_linux && \
    cmake .. && \
    make && \
    make install 
COPY requirements.txt /home
RUN pip3 install cython numpy && \
    pip3 install git+https://github.com/gregorias/pycodec2.git --global-option=build_ext --global-option="-I/usr/local/include/" --global-option="-L/usr/local/lib"  && \
    pip3 install -r /home/requirements.txt

