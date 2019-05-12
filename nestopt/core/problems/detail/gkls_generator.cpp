#include "nestopt/core/problems/detail/gkls_generator.hpp"

#include <new>
#include "nestopt/core/problems/detail/gkls_lib.h"

namespace nestopt {
namespace core {
namespace problems {
namespace details {

GKLSGenerator::GKLSGenerator() {
  TryThrow( gkls_create(&gkls_) );
}

GKLSGenerator::GKLSGenerator(const GKLSGenerator &other) {
  TryThrow( gkls_copy(other.gkls_, &gkls_) );
}

GKLSGenerator::GKLSGenerator(GKLSGenerator &&other) noexcept
  : gkls_(nullptr) { swap(other); }

GKLSGenerator::~GKLSGenerator() {
  Free();
}

GKLSGenerator &GKLSGenerator::operator = (GKLSGenerator other) {
  return swap(other);
}

GKLSGenerator &GKLSGenerator::operator = (GKLSGenerator &&other) noexcept {
  swap(other);
  other.Free();
  return *this;
}

GKLSGenerator &GKLSGenerator::Generate() {
  return TryThrow( gkls_generate(gkls_) );
}

GKLSGenerator &GKLSGenerator::dimension(size_t value) {
  return TryThrow(gkls_set_dim(gkls_, value));
}

GKLSGenerator &GKLSGenerator::minima_count(size_t value) {
  return TryThrow( gkls_set_num_minima(gkls_, value) );
}

GKLSGenerator &GKLSGenerator::function_number(int value) {
  return TryThrow( gkls_set_function_number(gkls_, (unsigned)value) );
}

GKLSGenerator &GKLSGenerator::global_dist(double value) {
  return TryThrow( gkls_set_global_dist(gkls_, value) );
}

GKLSGenerator &GKLSGenerator::global_value(double value) {
  return TryThrow( gkls_set_global_value(gkls_, value) );
}

GKLSGenerator &GKLSGenerator::global_radius(double value) {
  return TryThrow( gkls_set_global_radius(gkls_, value) );
}

GKLSGenerator &GKLSGenerator::domain_left(const double *left, size_t size) {
  return TryThrow( gkls_set_domain_left(gkls_, left) );
}

GKLSGenerator &GKLSGenerator::domain_right(const double *right, size_t size) {
  return TryThrow( gkls_set_domain_right(gkls_, right) );
}

void GKLSGenerator::global_minimizer(size_t index, double *x, size_t size) const {
  TryThrow( gkls_get_global_minimizer(gkls_, index, x) );
}

size_t GKLSGenerator::global_minimizers_count() const {
  size_t count;
  TryThrow( gkls_get_global_minimizers_num(gkls_, &count) );
  return count;
}

double GKLSGenerator::CalculateND(const double *x, size_t size) const {
  double f;
  TryThrow( gkls_calculate_nd(gkls_, x, &f) );
  return f;
}

double GKLSGenerator::CalculateD(const double *x, size_t size) const {
  double f;
  TryThrow( gkls_calculate_d(gkls_, x, &f) );
  return f;
}

double GKLSGenerator::CalculateD2(const double *x, size_t size) const {
  double f;
  TryThrow( gkls_calculate_d2(gkls_, x, &f) );
  return f;
}

GKLSGenerator &GKLSGenerator::swap(GKLSGenerator &other) noexcept {
  auto tmp = gkls_;
  gkls_ = other.gkls_;
  other.gkls_ = tmp;
  return *this;
}

GKLSGenerator &GKLSGenerator::TryThrow(gkls_error error) const {
  if (error != GKLS_OK) {
    if (error == GKLS_MEMORY_ERROR)
      throw std::bad_alloc();

    const char *description = NULL;
    gkls_get_error_description(error, &description);
    throw GKLSException(error, description);
  }
  return const_cast<GKLSGenerator &>(*this);
}

void GKLSGenerator::Free() {
  if (gkls_) {
    TryThrow( gkls_free(gkls_) );
  }
  gkls_ = nullptr;
}

}  // namespace details
}  // namespace problems
}  // namespace core
}  // namespace nestopt
