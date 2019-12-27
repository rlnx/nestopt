#pragma once

#include <bitset>
#include <ostream>
#include <queue>

#include "nestopt/core/types.hpp"
#include "nestopt/core/utils/common.hpp"

namespace nestopt {
namespace core {
namespace direct {

inline Scalar GetCubeDelta(Size round) {
  return std::pow(Scalar(3.), -Scalar(round));
}

inline Size GetCubeIndex(Size dimension, Size round, Size used_axis_count) {
  NestoptAssert(used_axis_count < dimension);
  return round * dimension + used_axis_count;
}

inline Scalar GetCubeDiagonal(Size dimension, Size round, Size used_axis_count) {
  const Scalar delta = GetCubeDelta(round);
  const Scalar factor = Scalar(8. / 9.);
  return delta * std::sqrt(Scalar(dimension) - factor * Scalar(used_axis_count));
}

class Cube;
class CubeSet;

class Cube {
 public:
  static constexpr Size max_cube_dimension = 32;
  using AxesBitset = std::bitset<max_cube_dimension>;

  explicit Cube(const Vector &x, Scalar z)
      : x_(x), z_(z) {
    NestoptAssert(x.size() > 0);
  }

  explicit Cube(const Vector &x, Scalar z,
                Size round, Size used_axis_count,
                const AxesBitset &used_axes)
      : x_(x),
        z_(z),
        round_(round),
        used_axis_count_(used_axis_count),
        used_axes_(used_axes) {
    NestoptAssert(x.size() > 0);
    NestoptAssert(used_axis_count < x.size());
    NestoptAssert(used_axis_count == used_axes.count());
  }

  void Split(CubeSet &output_container, const Objective &function) const;

  Size dimension() const { return x_.size(); }

  Size round() const { return round_; }

  Size used_axis_count() const { return used_axis_count_; }

  const Vector &x() const { return x_; }

  Scalar z() const { return z_; }

  auto &used_axes() const { return used_axes_; }

  Size index() const {
    return GetCubeIndex(dimension(), round(), used_axis_count());
  }

  Scalar diag() const {
    return GetCubeDiagonal(dimension(), round(), used_axis_count());
  }

 private:
  Vector x_;
  Scalar z_ = utils::Infinity();
  Size round_ = 0;
  Size used_axis_count_ = 0;
  AxesBitset used_axes_;
};

inline std::ostream &operator <<(std::ostream &stream, const Cube &cube) {
  stream << "Cube { "
         << "(" << cube.round() << ", " << cube.used_axis_count() << "), "
         << "x = " << cube.x() << ", "
         << "z = " << cube.z() << " }";
  return stream;
}

class CubeGroup {
 public:
  CubeGroup() = default;
  CubeGroup(const CubeGroup &other) = delete;
  CubeGroup &operator=(const CubeGroup &other) = delete;

  void push_back(const Cube &cube) {
    queue_.emplace(new Cube{cube});
  }

  template <typename... Args>
  void emplace_back(Args &&... args) {
    queue_.emplace(new Cube{std::forward<Args>(args)...});
  }

  bool empty() const {
    return queue_.empty();
  }

  Cube top() const {
    return *queue_.top().cube;
  }

  void pop() {
    queue_.pop();
  }

 private:
  struct Entry {
    explicit Entry(Cube *cube)
      : cube(cube),
        priority(cube->z()) {}

    bool operator <(const Entry &other) const {
      return priority > other.priority;
    }

    Cube *cube;
    Scalar priority;
  };

  struct Container : public std::vector<Entry> {
    Container() = default;
    Container(const Container &other) = delete;
    Container &operator=(const Container &other) = delete;

    ~Container() {
      for (auto &entry : *this) {
        delete entry.cube;
        entry.cube = nullptr;
      }
    }
  };

  std::priority_queue<Entry, Container> queue_;
};

class CubeSet {
 public:
  explicit CubeSet(Size dimension, Size expected_rounds) {
    groups_.reserve(dimension * expected_rounds + dimension);
  }

  CubeSet(CubeSet &&other) = default;
  CubeSet(const CubeSet &other) = delete;
  CubeSet &operator =(const CubeSet &other) = delete;

  ~CubeSet() {
    for (CubeGroup *group : groups_) {
      delete group;
      group = nullptr;
    }
  }

  void push_back(const Cube &cube) {
    get_group(cube.index()).push_back(cube);
  }

  void emplace_back(const Vector &x, Scalar z,
                    Size round, Size used_axis_count,
                    const Cube::AxesBitset &used_axes) {
    const Size index = GetCubeIndex(x.size(), round, used_axis_count);
    get_group(index).emplace_back(x, z, round, used_axis_count, used_axes);
  }

  bool pop(Size index) {
    const bool can_pop = index < groups_.size() &&
                         groups_[index] &&
                         !groups_[index]->empty();
    if (can_pop) {
      groups_[index]->pop();
    }
    return can_pop;
  }

  void pop_all() {
    for (CubeGroup *group : groups_) {
      if (group && !group->empty()) {
        group->pop();
      }
    }
  }

  std::vector<Cube> top() const {
    std::vector<Cube> result;
    result.reserve(groups_.size());

    const std::int64_t n = groups_.size();
    for (std::int64_t i = n - 1; i >= 0; i--) {
      if (groups_[i] && !groups_[i]->empty()) {
        result.push_back(groups_[i]->top());
      }
    }

    return result;
  }

  bool empty() const {
    for (CubeGroup *group : groups_) {
      if (group && !group->empty()) {
        return false;
      }
    }
    return true;
  }

 private:
  CubeGroup &get_group(Size index) {
    if (groups_.size() <= index) {
      groups_.resize(index + 1);
    }
    if (!groups_[index]) {
      groups_[index] = new CubeGroup();
    }
    return *groups_[index];
  }

  std::vector<CubeGroup *> groups_;
};

}  // namespace direct
}  // namespace core
}  // namespace nestopt
