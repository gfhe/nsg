//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

/**
 * 加载数据，生成data，数据点数，每个点的维度数
 * 
 * 数据文件格式：？
 * 
 * filename: 输入的数据文件
 * data：获取加载后的数据
 * num：获取数据中点的个数
 * dim：获取每个数据点的维度
*/
void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // 获取数据文件的长度
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;

  // 数据中的点个数， dim+1表示每个数据后面有一个分隔符？
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  // 从头将数据文件中的点数据读入data
  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}
int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0] << " data_file nn_graph_path L R C save_graph_file"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

  auto s = std::chrono::high_resolution_clock::now();
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);

  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  index.Save(argv[6]);

  return 0;
}
