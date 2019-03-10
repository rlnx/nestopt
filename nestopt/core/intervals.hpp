#ifndef NESTOPT_CORE_INTERVALS_HPP_
#define NESTOPT_CORE_INTERVALS_HPP_

#include "nestopt/core/types.hpp"
#include "nestopt/core/utils/common.hpp"

namespace nestopt {
namespace core {

template<Size SIZE>
class IntervalSet {
public:
  explicit IntervalSet(Scalar reliability)
      : n_(0), t_(0), m_(0),
        r_(reliability),
        min_z_(utils::Infinity()) {
    NestoptAssert( r_ > 1.0 );
  }

  Scalar PushFirst(Scalar xl, Scalar zl,
                   Scalar xr, Scalar zr) {
    NestoptAssert( n_ == 0 );
    NestoptAssert( r_ > 1.0 );
    xl_[0] = xl; zl_[0] = zl;
    xr_[0] = xr; zr_[0] = zr;
    n_++;

    m_ = NormalizeM(Slope(0));
    UpdateWeight(0);

    min_z_ = utils::Min(zl, zr);
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

} // namespace core
} // namespace nestopt

#endif // NESTOPT_CORE_INTERVALS_HPP_
