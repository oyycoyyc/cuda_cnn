#include "dataset.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " dataset.bin\n";
    return 2;
  }

  try {
    const MnistDataset dataset = LoadMnistDataset(argv[1]);
    std::uint64_t checksum = 0;
    for (const std::uint8_t pixel : dataset.images) {
      checksum += pixel;
    }
    std::cout << "count=" << dataset.sample_count << " rows=" << dataset.rows
              << " columns=" << dataset.columns << " checksum=" << checksum
              << " labels=";
    for (std::size_t index = 0; index < dataset.labels.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << static_cast<unsigned int>(dataset.labels[index]);
    }
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
