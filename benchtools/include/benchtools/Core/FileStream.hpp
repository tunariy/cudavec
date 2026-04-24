#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace benchtools {

struct File {
  //
};

class FileStream {
public:
  explicit FileStream(std::string_view path);

  explicit FileStream() = default;

  void append(const std::string& string);

  void clear();

private:
  std::fstream mStream;
  std::filesystem::path mPath;
};
}  // namespace benchtools