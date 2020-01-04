#pragma once

#include "nestopt/types.hpp"
#include "nestopt/utils/common.hpp"

namespace nestopt {

class Interval {
public:
  Interval(Scalar x_left, Scalar z_left,
           Scalar x_right, Scalar z_right)
    : x_left_(x_left),
      z_left_(z_left),
      x_right_(x_right),
      z_right_(z_right) { }

  Scalar x_left() const { return x_left_; }
  Scalar z_left() const { return z_left_; }
  Scalar x_right() const { return x_right_; }
  Scalar z_right() const { return z_right_; }

private:
  Scalar x_left_;
  Scalar z_left_;
  Scalar x_right_;
  Scalar z_right_;
};

template<Size SIZE>
class IntervalSet {
public:
  explicit IntervalSet(Scalar reliability)
      : n_(0), t_(0), m_(0),
        r_(reliability),
        min_z_(utils::Infinity()) {
    NestoptAssert( r_ > 1.0 );
  }

  Scalar Reset(const Interval &interval) {
    return Reset(std::vector<Interval>({ interval }));
  }

  Scalar Reset(const std::vector<Interval> &intervals) {
    NestoptAssert( intervals.size() > 0 );
    NestoptAssert( intervals.size() <= SIZE );

    Scalar m = 0;
    Scalar z = utils::Infinity();

    for (Size i = 0; i < intervals.size(); i++) {
      xl_[i] = intervals[i].x_left();
      zl_[i] = intervals[i].z_left();
      xr_[i] = intervals[i].x_right();
      zr_[i] = intervals[i].z_right();
      m = utils::Max(m, Slope(i));
      z = utils::Min(z, zl_[i], zr_[i]);
    }

    n_ = intervals.size();
    m_ = NormalizeM(m);
    min_z_ = z;

    UpdateWeights();
    return BestLength();
  }

  Scalar Push(Scalar x, Scalar z) {
    NestoptAssert( n_ > 0 );
    NestoptAssert( n_ < SIZE );
    NestoptAssert( xl_[t_] < x );
    NestoptAssert( xr_[t_] > x );

    xr_[n_] = xr_[t_];
    zr_[n_] = zr_[t_];
    xr_[t_] = xl_[n_] = x;
    zr_[t_] = zl_[n_] = z;
    n_++;

    const Scalar m = utils::Max(Slope(t_),
                                Slope(n_ - 1));
    const Scalar old_m = m_;
    m_ = utils::Max(m_, NormalizeM(m));

    if (old_m < m_) {
      UpdateWeights();
    }
    else {
      UpdateWeightsLite();
    }

    min_z_ = utils::Min(min_z_, z);
    return BestLength();
  }

  Scalar Next() const {
    const Scalar zdiff = zr_[t_] - zl_[t_];
    return 0.5 * (xr_[t_] + xl_[t_] - zdiff / m_);
  }

  Scalar Min() const {
    return min_z_;
  }

  Scalar BestWeight() const {
    return weights_[t_];
  }

  Scalar BestLength() const {
    return xr_[t_] - xl_[t_];
  }

  Size size() const {
    return n_;
  }

  bool empty() const {
    return size() == 0;
  }

private:
  void UpdateWeightsLite() {
    const Scalar old_w = BestWeight();
    const Scalar new_w = utils::Max(UpdateWeight(t_),
                                    UpdateWeight(n_ - 1));
    if (new_w > old_w) {
      t_ = ArgmaxWeight(t_, n_ - 1);
    }
    else {
      UpdateBestIdx();
    }
  }

  void UpdateWeights() {
    Scalar mw = utils::NegativeInfinity();
    for (Size i = 0; i < n_; i++) {
      const Scalar w = UpdateWeight(i);
      if (w > mw) { mw = w; t_ = i; }
    }
  }

  void UpdateBestIdx() {
    Scalar mw = utils::NegativeInfinity();
    for (Size i = 0; i < n_; i++) {
      const Scalar w = weights_[i];
      if (w > mw) { mw = w; t_ = i; }
    }
  }

  Scalar UpdateWeight(Size i) {
    const Scalar xdiff = m_ * (xr_[i] - xl_[i]);
    const Scalar zdiff = zr_[i] - zl_[i];
    weights_[i] = xdiff + (zdiff * zdiff) / xdiff
                        - 2 * (zr_[i] + zl_[i]);
    return weights_[i];
  }

  Scalar Slope(Size i) const {
    return utils::Abs( zr_[i] - zl_[i] ) /
                     ( xr_[i] - xl_[i] );
  }

  Size ArgmaxWeight(Size i, Size j) const {
    return (weights_[i] > weights_[j]) ? i : j;
  }

  Scalar NormalizeM(Scalar m) const {
    const Scalar eps = 1e-7;
    return (m < eps) ? 1.0 : m * r_;
  }

private:
  Size n_;
  Size t_;

  Scalar m_;
  Scalar r_;
  Scalar min_z_;

  Scalar xl_[SIZE];
  Scalar zl_[SIZE];
  Scalar xr_[SIZE];
  Scalar zr_[SIZE];

  Scalar weights_[SIZE];
};

constexpr Size DEFAULT_INTERVAL_SET_MAX_SIZE = 256;
using DefaultIntervalSet = IntervalSet<DEFAULT_INTERVAL_SET_MAX_SIZE>;

} // namespace nestopt
