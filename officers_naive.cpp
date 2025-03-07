#include <iostream>
#include <vector>
#include <cstdint>

bool is_rare(uint16_t x) {
  uint16_t mask = 0b10001; // Specific to officers
  return (__builtin_popcount(x & ~mask) % 2) == 0;
}

int main() {
  std::vector<uint16_t> values;
  values.push_back(0); // G(0) = 0, special case because you can't remove a coin
  values.push_back(0); // G(1) = 0, special case because you can't split into 0 and 0
  
  for (uint64_t i = 2; i < 1000; i++) {
    std::vector<bool> seen(512, false);

    uint64_t midway = (i-1) >> 1;
    for (uint64_t j = 0; j <= midway; j++) {
      uint16_t value = values[j] ^ values[i - 1 - j];

      if(value >= seen.size()) // Won't ever happen in practice
        seen.resize(value + 1, false);

      seen[value] = true;
    }

    uint16_t mex = 0;
    while (mex < seen.size() && seen[mex]) {
      mex++;
    }

    if(is_rare(mex))
      std::cout << "Rare: G(" << i << ") = " << mex << std::endl;
    else
      std::cout << "Comm: G(" << i << ") = " << mex << std::endl;

    values.push_back(mex);
  }
  return 0;
}


// Even more naive:

// #include <iostream>
// #include <vector>
// #include <unordered_set>

// int main() {
//   std::vector<int> values;

//   for (int i = 0; i < 1000; i++) {
//     std::unordered_set<int> seen;

//     for (int j = 1; j < i; j++) {
//       if(j + j != i)
//         seen.insert(values[j] ^ values[i - j]);
//     }

//     int mex = 0;
//     while (seen.count(mex)) mex++;

//     values.push_back(mex);
//     std::cout << mex << '\n';
//   }
// }
