FROM ubuntu:jammy

RUN apt-get update
RUN apt-get upgrade
RUN apt -y install build-essential git cmake

RUN apt-get -y install software-properties-common
RUN add-apt-repository universe
RUN apt-get -y install qt6-base-dev
RUN apt-get -y install libgl1-mesa-dev libglvnd-dev
RUN apt-get -y install libinsighttoolkit4-dev

RUN apt-get -y install libceres-dev

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt install -y tzdata

RUN apt-get -y install libvtk9-dev
RUN apt-get -y install qtbase5-dev
RUN apt-get -y install xtensor-dev
RUN apt-get -y install libopencv-dev
RUN apt-get -y install libspdlog-dev
RUN apt-get -y install doxygen
RUN apt-get -y install libboost-system-dev libboost-program-options-dev
RUN apt-get -y install libgsl-dev
RUN apt-get -y install libsdl2-dev

COPY . /src

RUN mkdir /build
WORKDIR /build

RUN cmake -DVC_BUILD_ACVD=on -DVC_BUILD_JSON=on -DVC_BUILD_DOCS=off -DVC_WITH_CUDA_SPARSE=off /src
RUN make -j16