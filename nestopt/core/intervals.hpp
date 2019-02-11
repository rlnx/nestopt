#ifndef NESTOPT_CORE_INTERVALS_HPP
#define NESTOPT_CORE_INTERVALS_HPP

#include "nestopt/core/types.hpp"

namespace nestopt {
namespace core {

class IntervalSet {
public:
  explicit IntervalSet(scalar_t reliability)
      : n_(0), t_(0), m_(0),
        r_(reliability),
        min_z_(utils::infty()) {
    _ADAPTIVE_AGS_ASSERT_( r_ > 1.0 );
  }

  scalar_t push_first(scalar_t xl, scalar_t zl,
                      scalar_t xr, scalar_t zr) {
    _ADAPTIVE_AGS_ASSERT_( n_ == 0 );
    _ADAPTIVE_AGS_ASSERT_( r_ > 1.0 );
    xl_[0] = xl; zl_[0] = zl;
    xr_[0] = xr; zr_[0] = zr;
    n_++;

    m_ = normalize_m(slope(0));
    update_weight(0);

    min_z_ = utils::min(zl, zr);
    return best_length();
  }

  scalar_t push(scalar_t x, scalar_t z) {
    _ADAPTIVE_AGS_ASSERT_( n_ > 0 );
    _ADAPTIVE_AGS_ASSERT_( n_ < SIZE );
    _ADAPTIVE_AGS_ASSERT_( xl_[t_] < x );
    _ADAPTIVE_AGS_ASSERT_( xr_[t_] > x );

    xr_[n_] = xr_[t_];
    zr_[n_] = zr_[t_];
    xr_[t_] = xl_[n_] = x;
    zr_[t_] = zl_[n_] = z;

    const scalar_t m = utils::max(slope(t_),
                                  slope(n_));
    m_ = utils::max(m_, normalize_m(m));

    n_++;

    // if (m > m_) {
      // m_ = normalize_m(m);
      update_weights();
    // }
    // else {
      // update_weights_lite();
    // }

    min_z_ = utils::min(min_z_, z);
    return best_length();
  }

  bool update(scalar_t x, scalar_t z) {
    size_t updated_idx[2];
    char updated_idx_counter = 0;

    scalar_t mw = utils::neg_infty();
    for (size_t i = 0; i < n_; i++) {
      const bool l = (xl_[i] == x);
      const bool r = (xr_[i] == x);
      if (l) { zl_[i] = z; }
      if (r) { zr_[i] = z; }
      if (l || r) {
        _ADAPTIVE_AGS_ASSERT_( updated_idx_counter < 2 );
        updated_idx[updated_idx_counter++] = i;
      }
    }
    _ADAPTIVE_AGS_ASSERT_(updated_idx_counter <= 2);

    scalar_t m = utils::neg_infty();
    for (char i = 0; i < updated_idx_counter; i++) {
      const size_t j = updated_idx[i];
      m = utils::max(m, slope(j));
    }

    if (updated_idx_counter > 0) {
      m_ = utils::max(m_, normalize_m(m));
      update_weights();
    }

    min_z_ = utils::min(min_z_, z);
    return updated_idx_counter > 0;
  }

  scalar_t next() const {
    const scalar_t zdiff = zr_[t_] - zl_[t_];
    return 0.5 * (xr_[t_] + xl_[t_] - zdiff / m_);
  }

  scalar_t min_z() const {
    return min_z_;
  }

  size_t size() const {
    return n_;
  }

  scalar_t best_weight() const {
    return weights_[t_];
  }

  scalar_t best_length() const {
    return xr_[t_] - xl_[t_];
  }

  bool empty() const {
    return size() == 0;
  }

private:
  void update_weights_lite() {
    const scalar_t old_w = best_weight();
    const scalar_t new_w = utils::max(update_weight(t_),
                                      update_weight(n_ - 1));
    if (new_w > old_w) {
      t_ = argmax_weight(t_, n_ - 1);
    }
    else {
      update_t();
    }
  }

  void update_weights() {
    scalar_t mw = utils::neg_infty();
    for (size_t i = 0; i < n_; i++) {
      const scalar_t w = update_weight(i);
      if (w > mw) { mw = w; t_ = i; }
    }
  }

  void update_t() {
    scalar_t mw = utils::neg_infty();
    for (size_t i = 0; i < n_; i++) {
      const scalar_t w = weights_[i];
      if (w > mw) { mw = w; t_ = i; }
    }
  }

  scalar_t update_weight(size_t i) {
    const scalar_t xdiff = m_ * (xr_[i] - xl_[i]);
    const scalar_t zdiff = zr_[i] - zl_[i];
    weights_[i] = xdiff + (zdiff * zdiff) / xdiff
                        - 2 * (zr_[i] + zl_[i]);
    return weights_[i];
  }

  scalar_t slope(size_t i) const {
    return utils::abs( zr_[i] - zl_[i] ) /
                     ( xr_[i] - xl_[i] );
  }

  size_t argmax_weight(size_t i, size_t j) const {
    return (weights_[i] > weights_[j]) ? i : j;
  }

  scalar_t normalize_m(scalar_t m) const {
    const scalar_t eps = 1e-7;
    return (m < eps) ? 1.0 : m * r_;
  }

private:
  size_t n_;
  size_t t_;

  scalar_t m_;
  scalar_t r_;
  scalar_t min_z_;

  scalar_t xl_[SIZE];
  scalar_t zl_[SIZE];
  scalar_t xr_[SIZE];
  scalar_t zr_[SIZE];

  scalar_t weights_[SIZE];
};

} // namespace core
} // namespace nestopt

#endif // NESTOPT_CORE_INTERVALS_HPP
