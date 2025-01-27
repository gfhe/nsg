// 邻居节点 数据结构
// Neighbor：NSG算法使用的邻居
// nhood ：knn图的节点表示 


#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <vector>
#include <mutex>

namespace efanna2e {

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

typedef std::lock_guard<std::mutex> LockGuard;
struct nhood{
  std::mutex lock;
  std::vector<Neighbor> pool;
  unsigned M;

  std::vector<unsigned> nn_old;
  std::vector<unsigned> nn_new;
  std::vector<unsigned> rnn_old;
  std::vector<unsigned> rnn_new;

  nhood(){}
  nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N){
    M = s;
    nn_new.resize(s * 2);
    GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
    nn_new.reserve(s * 2);
    pool.reserve(l);
  }

  nhood(const nhood &other){
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
  }
  /**
   * 往nhood中插入节点，并且保证当前点的邻居节点数量固定为 l。
   * 待插入点id，待插入点与当前点的距离dist
  */
  void insert (unsigned id, float dist) {
    LockGuard guard(lock);
    if (dist > pool.front().distance) return;
    // 检查是否已经存在
    for(unsigned i=0; i<pool.size(); i++){
      if(id == pool[i].id)return;
    }
    // 直接插入情况
    if(pool.size() < pool.capacity()){
      pool.push_back(Neighbor(id, dist, true));
      std::push_heap(pool.begin(), pool.end());
    }else{
      // 插入前移除节点（堆为最大堆）。
      std::pop_heap(pool.begin(), pool.end()); // 移动堆顶元素到数组末尾
      pool[pool.size()-1] = Neighbor(id, dist, true); // 覆盖插入节点
      std::push_heap(pool.begin(), pool.end()); // 重构最大堆
    }

  }

  template <typename C>
  void join (C callback) const {
    for (unsigned const i: nn_new) {
      for (unsigned const j: nn_new) {
        if (i < j) {
          callback(i, j);
        }
      }
      for (unsigned j: nn_old) {
        callback(i, j);
      }
    }
  }
};

struct SimpleNeighbor{
  unsigned id;
  float distance;

  SimpleNeighbor() = default;
  SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

  inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
  }
};
struct SimpleNeighbors{
  std::vector<SimpleNeighbor> pool;
};

/**
 * 有序待选集合插入操作：将数据插入到待选集合，并移除不符合要求的点
 * 返回插入的位置
 * 
 * 问题：当待插入点距离 比最小的待选集合距离还小，直接覆盖最小点是否合理？不应该是所有元素右移吗？
 * 
 * addr：待选集合内存地址
 * K：待选集合大小
 * nn: 待插入的点
 * 
*/
static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left=0,right=K-1;
  
  // 待插入点距离 比最小的还小，则直接覆盖
  if(addr[left].distance>nn.distance){
    memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  
  // 待插入点距离 比最大的还大，则直接覆盖到位置K
  if(addr[right].distance<nn.distance){
    addr[K] = nn;
    return K;
  }

  // 待插入点距离 在 left 到right 间，则二分查找替换点
  while(left<right-1){
    int mid=(left+right)/2;
    if(addr[mid].distance>nn.distance)right=mid;
    else left=mid;
  }
  //check equal ID

  while (left > 0){
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
  memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
  addr[right]=nn;
  return right;
}

}

#endif //EFANNA2E_GRAPH_H
