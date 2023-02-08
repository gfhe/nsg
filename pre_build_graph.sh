mkdir data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzf gist.tar.gz
wget -P data http://downloads.zjulearning.org.cn/nsg/sift_200nn.graph
wget -P data http://downloads.zjulearning.org.cn/nsg/gist_400nn.graph