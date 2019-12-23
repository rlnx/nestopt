#include <iostream>

#include "gtest/gtest.h"

#include "nestopt/core/intervals.hpp"
#include "nestopt/core/utils/common.hpp"
#include "nestopt/core/utils/random.hpp"

namespace nestopt {
namespace core {
namespace {

template<typename FloatType>
inline FloatType default_weight_function(FloatType m,
  FloatType l, FloatType r, FloatType lvalue, FloatType rvalue) {
  FloatType xdiff = r - l;
  FloatType zdiff = rvalue - lvalue;
  return m * xdiff + (zdiff * zdiff) / (m * xdiff) - 2.0 * (rvalue + lvalue);
}

template<typename FloatType>
inline FloatType default_next_function(FloatType m,
  FloatType l, FloatType r, FloatType lvalue, FloatType rvalue) {
  return 0.5 * (r + l - (rvalue - lvalue) / m);
}

class RefIntervalSet {
private:
  Size _t;
  Size _size;
  Size _capacity;

  Scalar _r;
  Scalar _m;
  Scalar _tv;
  Vector _xx;
  Vector _zz;

public:
  explicit RefIntervalSet(Scalar reliability, Size capacity)
    : _t(0),
      _size(0),
      _capacity(capacity + 1),
      _r(reliability),
      _m(utils::NegativeInfinity()),
      _tv(utils::NegativeInfinity()),
      _xx(Vector::Empty(capacity + 1)),
      _zz(Vector::Empty(capacity + 1)) { }

  void Reset(const Interval &interval) {
    return PushFirst(interval.x_left(), interval.z_left(),
                     interval.x_right(), interval.z_right());
  }

  void PushFirst(Scalar l, Scalar lvalue,
                 Scalar r, Scalar rvalue) {
    NestoptAssert( _size == 0 );
    NestoptAssert( _capacity > 0 );

    _xx[0] = l; _zz[0] = lvalue;
    _xx[1] = r; _zz[1] = rvalue;
    _m = _r * utils::Abs((rvalue - lvalue) / (r - l));
    _t = 1;
    _tv = default_weight_function(_m, _xx[0], _xx[1], _zz[0], _zz[1]);
    _size = 2;
  }

  Scalar Push(Scalar x, Scalar z) {
    NestoptAssert( _size < _capacity );
    _size++;

    for (Size i = _size - 1; i > 0; i--) {
      if (_xx[i - 1] < x) {
        _xx[i] = x;
        _zz[i] = z;
        break;
      }

      _xx[i] = _xx[i - 1];
      _zz[i] = _zz[i - 1];
    }

    Scalar mslope = utils::NegativeInfinity();
    for (Size i = 1; i < _size; i++) {
      Scalar s = utils::Abs((_zz[i] - _zz[i - 1]) / (_xx[i] - _xx[i - 1]));
      mslope = utils::Max(mslope, s);
    }
    _m = _r * mslope;

    _tv = utils::NegativeInfinity();
    for (Size i = 1; i < _size; i++) {
      Scalar c = default_weight_function(_m,
        _xx[i - 1], _xx[i], _zz[i - 1], _zz[i]);
      if (c > _tv) {
        _tv = c;
        _t = i;
      }
    }

    return _xx[_t] - _xx[_t - 1];
  }

  Scalar Next() const {
    return default_next_function(_m,
      _xx[_t - 1], _xx[_t], _zz[_t - 1], _zz[_t]);
  }

  Scalar BestWeight() const {
    return _tv;
  }

  const Vector &Arguments() const {
    return _xx;
  }

  const Vector &Values() const {
    return _zz;
  }
};

template<typename IntervalSetType>
Vector GenerateSequence(IntervalSetType &&iset, Size trials_num) {
  constexpr int seed = 77777;
  constexpr Scalar interval_beg = -1.5;
  constexpr Scalar interval_end =  1.0;
  constexpr Scalar value_min = -10.0;
  constexpr Scalar value_max =  10.0;

  std::mt19937 random_engine(seed);
  auto init_values = utils::GenerateUniform(random_engine, 2,
                                            value_min, value_max);
  iset.Reset({ interval_beg, init_values[0],
              interval_end, init_values[1] });

  auto values = utils::GenerateUniform(random_engine, trials_num,
                                       value_min, value_max);
  return Vector::Empty(trials_num).Fill([&] (Size i) {
    const Scalar x = iset.Next();
    const Scalar z = values[i];
    iset.Push(x, z);
    return x;
  });
}

TEST(Intervals, CompareWithReference) {
  constexpr Size trials_num = 200;
  constexpr Scalar reliability = 3.5;

  auto actual = GenerateSequence(DefaultIntervalSet(reliability), trials_num);
  auto ref = GenerateSequence(RefIntervalSet(reliability, trials_num + 2), trials_num);

  utils::Zip(actual, ref).ForEach([&] (Scalar x, Scalar y) {
    ASSERT_FLOAT_EQ(x, y);
  });
}

} // namespace
} // namespace core
} // namespace nestopt
