#ifndef NESTOPT_CORE_PROBLEMS_DETAIL_GKLSLIB_H_
#define NESTOPT_CORE_PROBLEMS_DETAIL_GKLSLIB_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GKLS_PI 3.14159265

#define GKLS_MAX_VALUE        1E+100
#define GKLS_PRECISION        1.0E-10
#define GKLS_PARABOLOID_MIN   0.0
#define GKLS_GLOBAL_MIN_VALUE -1.0
#define GKLS_DELTA_MAX_VALUE  10.0

#define GKLS_OK                                  0
#define GKLS_DIM_ERROR                           1
#define GKLS_NUM_MINIMA_ERROR                    2
#define GKLS_FUNC_NUMBER_ERROR                   3
#define GKLS_BOUNDARY_ERROR                      4
#define GKLS_GLOBAL_MIN_VALUE_ERROR              5
#define GKLS_GLOBAL_DIST_ERROR                   6
#define GKLS_GLOBAL_RADIUS_ERROR                 7
#define GKLS_MEMORY_ERROR                        8
#define GKLS_DERIV_EVAL_ERROR                    9
#define GKLS_GREAT_DIM                          10
#define GKLS_RHO_ERROR                          11
#define GKLS_PEAK_ERROR                         12
#define GKLS_GLOBAL_BASIN_INTERSECTION          13
#define GKLS_PARABOLA_MIN_COINCIDENCE_ERROR     14
#define GKLS_LOCAL_MIN_COINCIDENCE_ERROR        15
#define GKLS_FLOATING_POINT_ERROR               16
#define GKLS_NOT_INITIALIZED_ERROR              17
#define GKLS_ARGUMENT_OUTSIDE_OF_DOMAIN_ERROR   18
#define GKLS_NULL_POINTER_STATE_ERROR           19
#define GKLS_NULL_POINTER_ARGUMENT_ERROR        20
#define GKLS_INVALID_INDEX_ERROR                21
#define GKLS_INVALID_ERROR_CODE                 22

/**
 * @brief Type describes the GKLS error.
 */
typedef int gkls_error;

/**
 * @brief The GKLS generator state.
 */
typedef void *gkls_generator_state;

typedef const char *gkls_error_description;

/**
 * @brief Creates GKLS generator state.
 * @param[out] s Pointer to the state will be created.
 * @return GKLS error code.
 */
gkls_error gkls_create(gkls_generator_state *s);

/**
 * @brief Makes copy of the GKLS generator state.
 * @param[in] source GKLS generator source state.
 * @param[out] dest GKLS generator destination state.
 * @return GKLS error code.
 */
gkls_error gkls_copy(gkls_generator_state source, gkls_generator_state *dest);

/**
 * @brief Generates the GKLS function with given number.
 * @param[in] s GKLS generator state.
 * @return GKLS error code.
 */
gkls_error gkls_generate(gkls_generator_state s);

/**
 * @brief Releases the GKLS generator state.
 * @param[in] s GKLS generator state.
 * @return GKLS error code.
 */
gkls_error gkls_free(gkls_generator_state s);

/**
 * @brief Sets dimension of the GKLS function.
 * @param[in] s GKLS generator state.
 * @param[in] dim GKLS function dimension.
 * @return GKLS error code.
 */
gkls_error gkls_set_dim(gkls_generator_state s, size_t dim);

/**
 * @brief Sets the number of local minimizers of the GKLS function.
 * @param[in] s GKLS generator state.
 * @param[in] num Number of local minimizers.
 * @return GKLS error code.
 */
gkls_error gkls_set_num_minima(gkls_generator_state s, size_t num);

/**
 * @brief Sets the number of the GKLS function to generated.
 * @param[in] s GKLS generator state.
 * @param[in] number GKLS function number.
 * @return GKLS error code.
 */
gkls_error gkls_set_function_number(gkls_generator_state s, unsigned int number);

/**
 * @brief Sets the distance from the paraboloid minimizer to the global minimizer.
 * @param[in] s GKLS generator state.
 * @param[in] dist Distance from the paraboloid minimizer to the global minimizer.
 * @return GKLS error code.
 */
gkls_error gkls_set_global_dist(gkls_generator_state s, double dist);

/**
 * @brief Sets the global minimum value.
 * @param[in] s GKLS generator state.
 * @param[in] value Global minimum value.
 * @return GKLS error code.
 */
gkls_error gkls_set_global_value(gkls_generator_state s, double value);

/**
 * @brief Sets the radius of the global minimizer attraction region.
 * @param[in] s GKLS generator state.
 * @param[in] radius Radius of the global minimizer attraction region.
 * @return GKLS error code.
 */
gkls_error gkls_set_global_radius(gkls_generator_state s, double radius);

/**
 * @brief Sets the left boundary vector of the GKLS domain.
 * @param s[in] GKLS generator state.
 * @param left[in] Left boundary vector of the GKLS domain.
 * @return GKLS error code.
 */
gkls_error gkls_set_domain_left(gkls_generator_state s, const double *left);

/**
 * @brief Sets the right boundary vector of the GKLS domain.
 * @param s[in] GKLS generator state.
 * @param left[in] Right boundary vector of the GKLS domain.
 * @return GKLS error code.
 */
gkls_error gkls_set_domain_right(gkls_generator_state s, const double *right);

gkls_error gkls_get_global_minimizer(gkls_generator_state s, size_t i, double *minimizer);

gkls_error gkls_get_global_minimizers_num(gkls_generator_state s, size_t *num);

/**
 * @brief Calculates the non-differentiable GKLS function at the given point.
 * @param s[in] GKLS generator state.
 * @param x[in] Point to compute GKLS function value.
 * @param f[out] GKLS function value.
 * @return GKLS error code.
 */
gkls_error gkls_calculate_nd(gkls_generator_state s, const double *x, double *f);

/**
 * @brief Calculates the differentiable GKLS function at the given point.
 * @param s[in] GKLS generator state.
 * @param x[in] Point to compute GKLS function value.
 * @param f[out] GKLS function value.
 * @return GKLS error code.
 */
gkls_error gkls_calculate_d(gkls_generator_state s, const double *x, double *f);

/**
 * @brief Calculates the twice-differentiable GKLS function at the given point.
 * @param s[in] GKLS generator state.
 * @param x[in] Point to compute GKLS function value.
 * @param f[out] GKLS function value.
 * @return GKLS error code.
 */
gkls_error gkls_calculate_d2(gkls_generator_state s, const double *x, double *f);

gkls_error gkls_calculate_d_deriv(gkls_generator_state s, size_t i, const double *x, double *f);
gkls_error gkls_calculate_d2_deriv1(gkls_generator_state s, size_t i, const double *x, double *f);
gkls_error gkls_calculate_d2_deriv2(gkls_generator_state s, size_t i, const double *x, double *f);

gkls_error gkls_calculate_d_grad(gkls_generator_state s, size_t i, const double *x, double *f);
gkls_error gkls_calculate_d2_grad(gkls_generator_state s, size_t i, const double *x, double *f);
gkls_error gkls_calculate_d2_hess(gkls_generator_state s, size_t i, const double *x, double *f);

gkls_error gkls_get_error_description(gkls_error code, gkls_error_description *desc);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NESTOPT_CORE_PROBLEMS_DETAIL_GKLSLIB_H_
