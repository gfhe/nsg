sudo apt update -y
sudo apt -y install g++ cmake libboost-dev libgoogle-perftools-dev
mkdir build/ && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j