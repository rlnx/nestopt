#ifndef NESTOPT_CORE_TYPES_HPP_
#define NESTOPT_CORE_TYPES_HPP_

#include <vector>
#include <cstddef>
#include <utility>
#include <initializer_list>

#include <cstdio>

#ifdef _NDEBUG
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

enum class MemoryOwnership {
  self,
  user
};

template<typename T>
class Allocator {
public:
  T *allocate(Size size) {
    return new T[size];
  }

  void deallocate(const T *pointer) {
    delete[] pointer;
  }
};

class Vector {
public:
  typedef Scalar            value_type;
  typedef Allocator<Scalar> allocator_type;
  typedef Size              size_type;
  typedef std::ptrdiff_t    difference_type;
  typedef value_type&       reference;
  typedef const value_type& const_reference;
  typedef value_type*       pointer;
  typedef const pointer     const_pointer;

  static Vector &&Move(Vector &other) {
    return std::move(other);
  }

  static Vector View(const Vector &other) {
    return Vector(other.data(), other.size(),
                  MemoryOwnership::user);
  }

  static Vector Copy(const Vector &other) {
    return Vector(other.data(), other.size(),
                  MemoryOwnership::self);
  }

  static Vector View(pointer buffer, size_type size) {
    return Vector(buffer, size, MemoryOwnership::user);
  }

  static Vector Copy(const_pointer buffer, size_type size) {
    return Vector(const_cast<pointer>(buffer), size,
                  MemoryOwnership::self);
  }

  Vector()
    : data_(nullptr),
      size_(0),
      memowner_(MemoryOwnership::self) { }

  explicit Vector(size_type size)
      : data_(nullptr),
        size_(size),
        memowner_(MemoryOwnership::self) {
    data_ = allocator_type().allocate(size);
  }

  explicit Vector(pointer buffer,
                  size_type size,
                  MemoryOwnership owner)
      : data_(buffer),
        size_(size),
        memowner_(owner) {
    NestoptAssert(owner == MemoryOwnership::user ||
                  owner == MemoryOwnership::self);
    if (owner == MemoryOwnership::self) {
      data_ = CopyOwnData();
    }
  }

  Vector(const std::initializer_list<Scalar> &list)
      : Vector(list.size()) {
    size_type idx = 0;
    for (auto x : list) {
      data_[idx++] = x;
    }
  }

  Vector(const Vector &other) = delete;

  Vector(Vector &&other) :
      data_(other.data_),
      size_(other.size_),
      memowner_(other.memowner_) {
    other.memowner_ = MemoryOwnership::user;
    other.clean();
  }

  ~Vector() {
    clean();
  }

  Vector &operator=(const Vector &other) = delete;

  Vector &operator=(Vector &&other) {
    swap(other);
    other.clean();
    return *this;
  }

  size_type size() const {
    return size_;
  }

  pointer data() {
    return data_;
  }

  const_pointer data() const {
    return data_;
  }

  reference at(size_type i) {
    NestoptAssert(i < size_);
    return data_[i];
  }

  const_reference at(size_type i) const {
    NestoptAssert(i < size_);
    return data_[i];
  }

  reference operator[](size_type i) {
    return at(i);
  }

  const_reference operator[](size_type i) const {
    return at(i);
  }

  void clean() {
    if (memowner_ == MemoryOwnership::self) {
      allocator_type().deallocate(data_);
    }
    size_ = 0;
    data_ = nullptr;
  }

  Vector &swap(Vector &other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(memowner_, other.memowner_);
    return *this;
  }

private:
  pointer CopyOwnData() {
    pointer data_copy = allocator_type().allocate(size_);
    for (size_type i = 0; i < size_; i++) {
      data_copy[i] = data_[i];
    }
    return data_copy;
  }

  pointer data_;
  size_type size_;
  MemoryOwnership memowner_;
};

} // namespace nestopt
} // namespace core

#endif // NESTOPT_CORE_TYPES_HPP_
