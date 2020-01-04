#pragma once

#include <stddef.h>
#include <stdexcept>

namespace nestopt {
namespace problems {
namespace detail {

class GKLSException : public std::runtime_error {
public:
  explicit GKLSException(int code, const char *desc)
    : std::runtime_error(desc), gkls_error_code_(code) { }

  explicit GKLSException(int code, const std::string &desc)
    : std::runtime_error(desc), gkls_error_code_(code) { }

  int gkls_error_code() {
    return gkls_error_code_;
  }

private:
  int gkls_error_code_;
};

class GKLSGenerator {
public:
  GKLSGenerator();
  GKLSGenerator(const GKLSGenerator &);
  GKLSGenerator(GKLSGenerator &&) noexcept;
  ~GKLSGenerator();

  GKLSGenerator &operator = (GKLSGenerator);
  GKLSGenerator &operator = (GKLSGenerator &&) noexcept;

  GKLSGenerator &Generate();

  GKLSGenerator &dimension(size_t);
  GKLSGenerator &minima_count(size_t);
  GKLSGenerator &function_number(int);
  GKLSGenerator &global_dist(double);
  GKLSGenerator &global_value(double);
  GKLSGenerator &global_radius(double);
  GKLSGenerator &domain_left(const double *, size_t);
  GKLSGenerator &domain_right(const double *, size_t);

  void global_minimizer(size_t index, double *, size_t) const;
  size_t global_minimizers_count() const;

  double CalculateND(const double *, size_t) const;
  double CalculateD(const double *, size_t) const;
  double CalculateD2(const double *, size_t) const;

  GKLSGenerator &swap(GKLSGenerator &) noexcept;

private:
  GKLSGenerator &TryThrow(int error) const;
  void Free();

  void *gkls_;
};

} // namespace detail
} // namespace problems
} // namespace nestopt
