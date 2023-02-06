echo "build efanna"
cd ..
git clone https://github.com/ZJULearning/efanna_graph

cd efanna_graph
cd extern_libraries 

echo " install intel MLK" 
sudo apt -y install ncurses-term
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146_offline.sh
sudo sh ./l_BaseKit_p_2022.1.2.146_offline.sh

echo "build faiss"
git clone https://github.com/facebookresearch/faiss.git
cd faiss
echo "building faiss"
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON
make -C build -j faiss
echo "installing faiss headers and library"
make -C build install

cd ..

echo "to efanna root dir "
cd ..