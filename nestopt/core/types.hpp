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

template<typename T>
class DefaultAllocator {
public:
  T *allocate(Size size) {
    return new T[size];
  }

  void deallocate(const T *pointer) {
    delete[] pointer;
  }
};

template<typename T>
class Buffer {
public:
  virtual ~Buffer() { }
  virtual T *data() = 0;
  virtual Size size() = 0;
};
template<typename T>
using UniqueBuffer = std::unique_ptr<Buffer<T>>;

template<typename BufferType, typename ... Args>
std::unique_ptr<BufferType> MakeUniqueBuffer(Args &&...args) {
  return std::unique_ptr<BufferType>(new BufferType(std::forward<Args>(args)...));
}

template<typename T, typename Allocator = DefaultAllocator<T>>
class OwnBuffer : public Buffer<T> {
public:
  explicit OwnBuffer(Size size, const Allocator &allocator = Allocator())
    : allocator_(allocator),
      data_(allocator_.allocate(size)),
      size_(size) { }

  ~OwnBuffer() {
    allocator_.deallocate(data_);
  }

  T *data() override {
    return data_;
  }

  Size size() override {
    return size_;
  }

private:
  Allocator allocator_;
  T *data_;
  Size size_;
};

template<typename T>
class ViewBuffer : public Buffer<T> {
public:
  explicit ViewBuffer(T *data, Size size)
    : data_(data),
      size_(size) { }

  T *data() override {
    return data_;
  }

  Size size() override {
    return size_;
  }

 private:
  T *data_;
  Size size_;
};

template<typename T>
class VectorBase {
public:
  /* STL-compatible type traits */
  typedef T                 value_type;
  typedef Size              size_type;
  typedef std::ptrdiff_t    difference_type;
  typedef value_type&       reference;
  typedef const value_type& const_reference;
  typedef value_type*       pointer;
  typedef const pointer     const_pointer;

  VectorBase() = default;

  explicit VectorBase(T *data, Size size)
    : size_(size),
      data_(data) { }

  Size size() const {
    return size_;
  }

  T *data() {
    return data_;
  }

  const T *data() const {
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

private:
  Size size_ = 0;
  T *data_ = nullptr;
};

class Vector : public VectorBase<Scalar> {
public:
  typedef VectorBase<Scalar> super;

  static Vector Empty(Size size) {
    typedef OwnBuffer<Scalar> BufferType;
    return Vector(MakeUniqueBuffer<BufferType>(size));
  }

  static Vector Full(Size size, Scalar value) {
    return Empty(size).Fill(
      [&](Size i) { return value; }
    );
  }

  static Vector Zeros(Size size) {
    return Full(size, 0.0);
  }

  static Vector Copy(const std::vector<Scalar> &source) {
    return Empty(source.size()).Fill(
      [&](Size i) { return source[i]; }
    );
  }

  static Vector Wrap(Scalar *data, Size size) {
    return Vector(MakeUniqueBuffer<ViewBuffer<Scalar>>(data, size));
  }

  Vector() = default;

  explicit Vector(UniqueBuffer<Scalar> &&buffer)
    : super(buffer->data(), buffer->size()),
      buffer_(std::move(buffer)) { }

  Vector View(Size offset, Size subsize) const {
    NestoptAssert( offset + subsize <= size() );
    return Wrap(const_cast<Scalar *>(data() + offset), subsize);
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

private:
  UniqueBuffer<Scalar> buffer_;
};

} // namespace nestopt
} // namespace core

#endif // NESTOPT_CORE_TYPES_HPP_
