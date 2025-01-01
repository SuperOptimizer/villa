FROM ubuntu:jammy

RUN apt-get update
RUN apt-get upgrade
RUN apt -y install build-essential git cmake
RUN apt-get -y install software-properties-common
RUN add-apt-repository universe
RUN apt-get -y install qt6-base-dev
RUN apt-get -y install libgl1-mesa-dev libglvnd-dev
RUN apt-get -y install libceres-dev libboost-system-dev libboost-program-options-dev xtensor-dev libopencv-dev

COPY . /src

RUN mkdir /build
WORKDIR /build

RUN cmake -DVC_BUILD_JSON=on -DVC_WITH_CUDA_SPARSE=off /src
RUN make -j16