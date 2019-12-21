#ifndef NESTOPT_CORE_TYPES_HPP_
#define NESTOPT_CORE_TYPES_HPP_

#include <cstdio>
#include <cstddef>

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef NDEBUG
#define NESTOPT_DEBUG 0
#else
#define NESTOPT_DEBUG 1
#endif

#ifdef NESTOPT_DEBUG
#include <cassert>
#define NestoptAssert(expr) assert(expr)
#else
#define NestoptAssert(expr)
#endif

namespace nestopt {
namespace core {

using Scalar = double;
using Size   = std::size_t;

template <typename T>
using Shared = std::shared_ptr<T>;

template<typename T>
class VectorBase {
public:
  /* STL-compatible type traits */
  using value_type = T;
  using size_type = Size;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const pointer;

  VectorBase() = default;

  explicit VectorBase(const Shared<T> &base, T *data, Size size)
      : base_(base),
        data_(data),
        size_(size) {
      NestoptAssert(data >= base.get());
    }

  explicit VectorBase(const Shared<T> &data, Size size)
    : VectorBase(data, data.get(), size) {}

  template <typename Deleter = std::default_delete<T[]>>
  explicit VectorBase(T *base, T *data, Size size, Deleter &&deleter = Deleter{})
    : VectorBase(std::shared_ptr<T>(base, deleter), data, size) {}

  template <typename Deleter = std::default_delete<T[]>>
  explicit VectorBase(T *data, Size size, Deleter &&deleter = Deleter{})
    : VectorBase(std::shared_ptr<T>(data, deleter), data, size) {}

  Size size() const {
    return size_;
  }

  const T *data() const {
    return data_;
  }

  T *data() {
    return data_;
  }

  T &at(Size i) {
    NestoptAssert(i < size_);
    return data_[i];
  }

  const T &at(Size i) const {
    NestoptAssert(i < size_);
    return data_[i];
  }

  T &operator[](Size i) {
    return at(i);
  }

  const T &operator[](Size i) const {
    return at(i);
  }

protected:
  const Shared<T> &get_base() const {
    return base_;
  }

private:
  Shared<T> base_;
  T *data_ = nullptr;
  Size size_ = 0;
};

class Vector : public VectorBase<Scalar> {
public:
  using Super = VectorBase<Scalar>;
  using Super::Super;

  static Vector Empty(Size size) {
    return Vector(new Scalar[size], size);
  }

  static Vector Full(Size size, Scalar value) {
    return Empty(size).Fill([&](Size i) { return value; });
  }

  static Vector Zeros(Size size) {
    return Full(size, 0.0);
  }

  template <typename Container>
  static Vector Copy(const Container &source) {
    return Empty(source.size()).Fill([&](Size i) { return source[i]; });
  }

  static Vector Wrap(Scalar *data, Size size) {
    return Vector(data, size, [](Scalar *) {});
  }

  Vector View(Size offset, Size subsize) const {
    NestoptAssert(offset + subsize <= size());
    const auto data_mutable = const_cast<Scalar *>(data());
    return Vector(get_base(), data_mutable + offset, subsize);
  }

  template <typename Body>
  Vector &&For(const Body &body) {
    for (Size i = 0; i < size(); i++) {
      body(i);
    }
    return std::move(*this);
  }

  template <typename Body>
  Vector &&ForEach(const Body &body) {
    return For([&](Size i) { body(at(i)); });
  }

  template <typename Body>
  Vector &&Fill(const Body &body) {
    return For([&](Size i) { at(i) = body(i); });
  }
};

} // namespace nestopt
} // namespace core

#endif // NESTOPT_CORE_TYPES_HPP_
