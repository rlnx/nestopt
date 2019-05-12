#include "nestopt/core/problems/detail/gkls_lib.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#define GKLS_RAND_KK 100
#define GKLS_RAND_LL  37
#define GKLS_RAND_TT  70
#define GKLS_RAND_NUM_RND 1009

#define GKLS_IS_ODD(s)     ((s) & 1)
#define GKLS_MOD_SUM(x, y) ( ((x) + (y)) - (int)((x) + (y)) )

#define GKLS_MALLOC(type, size) (type*)malloc(sizeof(type) * (size))
#define GKLS_FREE(v)            free(v), v = NULL

#define GKLS_CHECK_CONDITION(cond, code) if (!(cond)) return code;
#define GKLS_THROW_ERROR(error)          if (error != GKLS_OK) return error;

typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct {
  size_t dim;
  size_t num_minima;
  uint number;
  double global_dist;
  double global_radius;
  double global_value;
  double *domain_left;
  double *domain_right;

  double **local_min;
  double *f;
  double *w_rho;
  double *peak;
  double *rho;
  double *rnd_num;
  double *ran_u;
  size_t num_global_minima;
  size_t *gm_index;

  double delta;
  ulong rnd_counter;

  int is_arg_set;
  int is_released;

  size_t allocated_dim;
  size_t allocated_num_minima;
  size_t allocated_left_domain_dim;
  size_t allocated_right_domain_dim;
} gkls_generator_state_internal;

typedef gkls_generator_state_internal *state_ptr;


double gkls_internal_norm(const double *x1, const double *x2, size_t dim) {
  size_t i;
  double norm = 0.0;
  for (i = 0; i < dim; i++)
    norm += (x1[i] - x2[i]) * (x1[i] - x2[i]);
  return sqrt(norm);
}

void gkls_internal_ranf_array(double *ran_u, double *aa, int n) {
  int i, j;

  for (j = 0; j < GKLS_RAND_KK; j++)
    aa[j] = ran_u[j];

  for (; j < n; j++)
    aa[j] = GKLS_MOD_SUM(aa[j - GKLS_RAND_KK], aa[j - GKLS_RAND_LL]);

  for (i = 0; i < GKLS_RAND_LL; i++, j++)
    ran_u[i] = GKLS_MOD_SUM(aa[j - GKLS_RAND_KK], aa[j - GKLS_RAND_LL]);

  for (; i < GKLS_RAND_KK; i++, j++)
    ran_u[i] = GKLS_MOD_SUM(aa[j - GKLS_RAND_KK], ran_u[i - GKLS_RAND_LL]);
}

void gkls_internal_ranf_start(double *ran_u, long seed) {
  int j;
  long t,s;
  double u[GKLS_RAND_KK + GKLS_RAND_KK - 1], ul[GKLS_RAND_KK + GKLS_RAND_KK - 1];
  double ulp = (1.0 / (1L << 30)) / (1L << 22);
  double ss = 2.0 * ulp * ((seed & 0x3fffffff) + 2);

  for (j = 0; j < GKLS_RAND_KK; j++) {
    u[j] = ss;
    ul[j] = 0.0;
    ss += ss;
    if (ss >= 1.0)
      ss -= 1.0 - 2 * ulp;
  }

  for (; j < GKLS_RAND_KK + GKLS_RAND_KK - 1; j++)
    u[j] = ul[j] = 0.0;

  u[1] += ulp;
  ul[1] = ulp;
  s = seed & 0x3fffffff;
  t = GKLS_RAND_TT - 1;
  while (t) {
    for (j = GKLS_RAND_KK - 1; j > 0; j--) {
      ul[j+j] = ul[j];
      u[j+j] = u[j];
    }

    for (j = GKLS_RAND_KK + GKLS_RAND_KK - 2; j > GKLS_RAND_KK - GKLS_RAND_LL; j -= 2) {
      ul[GKLS_RAND_KK + GKLS_RAND_KK - 1 - j] = 0.0;
      u[GKLS_RAND_KK + GKLS_RAND_KK - 1 - j] = u[j] - ul[j];
    }

    for (j = GKLS_RAND_KK + GKLS_RAND_KK - 2; j >= GKLS_RAND_KK; j--) {
      if (ul[j]) {
        ul[j - (GKLS_RAND_KK - GKLS_RAND_LL)] = ulp - ul[j - (GKLS_RAND_KK - GKLS_RAND_LL)],
        u[j - (GKLS_RAND_KK - GKLS_RAND_LL)] =
          GKLS_MOD_SUM(u[j - (GKLS_RAND_KK - GKLS_RAND_LL)], u[j]);
        ul[j - GKLS_RAND_KK] = ulp - ul[j - GKLS_RAND_KK];
        u[j - GKLS_RAND_KK] = GKLS_MOD_SUM(u[j - GKLS_RAND_KK], u[j]);
      }
    }

    if (GKLS_IS_ODD(s)) {
      for (j = GKLS_RAND_KK; j > 0; j--) {
        ul[j] = ul[j - 1];
        u[j] = u[j - 1];
      }

      ul[0] = ul[GKLS_RAND_KK];
      u[0] = u[GKLS_RAND_KK];
      if (ul[GKLS_RAND_KK]) {
        ul[GKLS_RAND_LL] = ulp - ul[GKLS_RAND_LL];
        u[GKLS_RAND_LL] = GKLS_MOD_SUM(u[GKLS_RAND_LL], u[GKLS_RAND_KK]);
      }
    }
    if (s) s >>= 1; else t--;
  }

  for (j = 0; j < GKLS_RAND_LL; j++)
    ran_u[j + GKLS_RAND_KK - GKLS_RAND_LL] = u[j];

  for (; j < GKLS_RAND_KK; j++)
    ran_u[j - GKLS_RAND_LL] = u[j];
}

void gkls_internal_initialize_rnd(state_ptr s, unsigned int nf) {
  long dim = (long)s->dim;
  long nmin = (long)s->num_minima;
  long seed = ((long)nf - 1L) + ((long)nmin - 1L) * 100L + (long)dim * 1000000L;
  gkls_internal_ranf_start(s->ran_u, seed);
}

gkls_error gkls_internal_alloc_left_bound(state_ptr s) {
  GKLS_FREE(s->domain_left);
  GKLS_CHECK_CONDITION(s->domain_left = GKLS_MALLOC(double, s->dim), GKLS_MEMORY_ERROR);
  s->allocated_left_domain_dim = s->dim;
  return GKLS_OK;
}

gkls_error gkls_internal_alloc_right_bound(state_ptr s) {
  GKLS_FREE(s->domain_right);
  GKLS_CHECK_CONDITION(s->domain_right = GKLS_MALLOC(double, s->dim), GKLS_MEMORY_ERROR);
  s->allocated_right_domain_dim = s->dim;
  return GKLS_OK;
}

gkls_error gkls_internal_alloc_bounds(state_ptr s) {
  gkls_error error;
  error = gkls_internal_alloc_left_bound(s); GKLS_THROW_ERROR(error);
  error = gkls_internal_alloc_right_bound(s); GKLS_THROW_ERROR(error);
  return GKLS_OK;
}

gkls_error gkls_internal_alloc(state_ptr s) {
  size_t i;

  if (s->local_min != NULL) {
    for (i = 0; i < s->num_minima; i++)
      GKLS_FREE(s->local_min[i]);
  }
  GKLS_FREE(s->local_min);

  GKLS_CHECK_CONDITION(s->local_min = GKLS_MALLOC(double*, s->num_minima), GKLS_MEMORY_ERROR);
  for (i = 0; i < s->num_minima; i++) {
    GKLS_CHECK_CONDITION(s->local_min[i] = GKLS_MALLOC(double, s->dim), GKLS_MEMORY_ERROR);
  }

  GKLS_FREE(s->w_rho);
  GKLS_CHECK_CONDITION(s->w_rho = GKLS_MALLOC(double, s->num_minima), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->peak);
  GKLS_CHECK_CONDITION(s->peak = GKLS_MALLOC(double, s->num_minima), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->rho);
  GKLS_CHECK_CONDITION(s->rho = GKLS_MALLOC(double, s->num_minima), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->f);
  GKLS_CHECK_CONDITION(s->f = GKLS_MALLOC(double, s->num_minima), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->rnd_num);
  GKLS_CHECK_CONDITION(s->rnd_num = GKLS_MALLOC(double, GKLS_RAND_NUM_RND), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->ran_u);
  GKLS_CHECK_CONDITION(s->ran_u = GKLS_MALLOC(double, GKLS_RAND_KK), GKLS_MEMORY_ERROR);

  GKLS_FREE(s->gm_index);
  GKLS_CHECK_CONDITION(s->gm_index = GKLS_MALLOC(size_t, s->num_minima), GKLS_MEMORY_ERROR);

  s->allocated_num_minima = s->num_minima;
  s->allocated_dim = s->dim;

  return GKLS_OK;
}

gkls_error gkls_internal_parameters_check(state_ptr s) {
  size_t i;
  double min_side, tmp;

  GKLS_CHECK_CONDITION(s->is_released == 0, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(s->dim > 1 && s->dim < GKLS_RAND_NUM_RND, GKLS_DIM_ERROR);
  GKLS_CHECK_CONDITION(s->num_minima > 1, GKLS_NUM_MINIMA_ERROR);
  GKLS_CHECK_CONDITION(s->domain_left != NULL, GKLS_BOUNDARY_ERROR);
  GKLS_CHECK_CONDITION(s->domain_right != NULL, GKLS_BOUNDARY_ERROR);

  for (i = 0; i < s->dim; i++) {
    tmp = s->domain_right[i] - s->domain_left[i];
    GKLS_CHECK_CONDITION(tmp > GKLS_PRECISION, GKLS_BOUNDARY_ERROR);
  }

  tmp = GKLS_PARABOLOID_MIN - s->global_value;
  GKLS_CHECK_CONDITION(s->global_value <= GKLS_PRECISION, GKLS_GLOBAL_MIN_VALUE_ERROR);

  min_side = s->domain_right[0] - s->domain_left[0];
  for (i = 1; i < s->dim; i++) {
    tmp = s->domain_right[i] - s->domain_left[i];
    if (tmp < min_side)
      min_side = tmp;
  }

  tmp = 0.5 * min_side - s->global_dist;
  GKLS_CHECK_CONDITION(tmp > GKLS_PRECISION, GKLS_GLOBAL_DIST_ERROR);
  GKLS_CHECK_CONDITION(s->global_dist > GKLS_PRECISION, GKLS_GLOBAL_DIST_ERROR);

  tmp = s->global_radius - 0.5 * s->global_dist;
  GKLS_CHECK_CONDITION(tmp < GKLS_PRECISION, GKLS_GLOBAL_RADIUS_ERROR);
  GKLS_CHECK_CONDITION(s->global_radius > GKLS_PRECISION, GKLS_GLOBAL_RADIUS_ERROR);

  return GKLS_OK;
}

gkls_error gkls_internal_coincidence_check(state_ptr s) {
  size_t i, j;
  double tmp_norm;

  for (i = 2; i < s->num_minima; i++) {
    tmp_norm = gkls_internal_norm(s->local_min[i], s->local_min[0], s->dim);
    GKLS_CHECK_CONDITION(tmp_norm >= GKLS_PRECISION, GKLS_PARABOLA_MIN_COINCIDENCE_ERROR);
  }

  for (i = 1; i < s->num_minima - 1; i++) {
    for (j = i + 1; j < s->num_minima; j++) {
      tmp_norm = gkls_internal_norm(s->local_min[i], s->local_min[j], s->dim);
      GKLS_CHECK_CONDITION(tmp_norm >= GKLS_PRECISION, GKLS_PARABOLA_MIN_COINCIDENCE_ERROR);
    }
  }

  return GKLS_OK;
}

gkls_error gkls_internal_set_basins(state_ptr s) {
  size_t i, j;
  double dist;
  double temp_min;
  double temp_d1, temp_d2;

  for (i = 0; i < s->num_minima; i++) {
    s->rho[i] = 0;
  }

  for (i = 0; i < s->num_minima; i++) {
    temp_min = GKLS_MAX_VALUE;
    for (j = 0; j < s->num_minima; j++) {
      if (i != j) {
        temp_d1 = gkls_internal_norm(s->local_min[i], s->local_min[j], s->dim);
        if (temp_d1 < temp_min)
          temp_min = temp_d1;
      }
    }
    dist = temp_min / 2.0;
    s->rho[i] = dist;
  }

  s->rho[1] = s->global_radius;
  for (i = 2; i < s->num_minima; i++) {
    dist = gkls_internal_norm(s->local_min[i], s->local_min[1], s->dim) - s->global_radius;
    if (dist - s->rho[i] < GKLS_PRECISION)
      s->rho[i] = dist;
  }

  for (i = 0; i < s->num_minima; i++) {
    if (i != 1) {
	    temp_min = GKLS_MAX_VALUE;
      for (j = 0; j < s->num_minima; j++) {
        if (i != j) {
          temp_d1 = gkls_internal_norm(s->local_min[i], s->local_min[j], s->dim) - s->rho[j];
          if (temp_d1 < temp_min)
            temp_min = temp_d1;
        }
      }
      if (temp_min - s->rho[i] > GKLS_PRECISION)
        s->rho[i] = temp_min;
    }
  }

  for (i = 0; i < s->num_minima; i++)
    s->rho[i] = s->w_rho[i] * s->rho[i];

  s->peak[0] = 0.0;
  s->peak[1] = 0.0;
  for (i = 2; i < s->num_minima; i++) {
    temp_d1 = gkls_internal_norm(s->local_min[0], s->local_min[i], s->dim);
    temp_min = (s->rho[i] - temp_d1) * (s->rho[i] - temp_d1) + s->f[0];
    temp_d1 = (1.0 + s->rnd_num[s->rnd_counter]) * s->rho[i];
    temp_d2 = s->rnd_num[s->rnd_counter] * (temp_min - s->global_value);

    if (temp_d2 < temp_d1)
      temp_d1 = temp_d2;
    s->peak[i]= temp_d1;

    s->rnd_counter++;
    if (s->rnd_counter == GKLS_RAND_NUM_RND) {
      gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);
      s->rnd_counter = 0L;
    }

    s->f[i] = temp_min - s->peak[i];
  }

  s->num_global_minima = 0;
  for (i = 0; i< s->num_minima; i++) {
    temp_d1 = s->global_value - s->f[i];
    temp_d2 = s->f[i] - s->global_value;

	  if (temp_d1 <= GKLS_PRECISION && temp_d2 <= GKLS_PRECISION) {
	     s->gm_index[s->num_global_minima] = i;
       s->num_global_minima++;
	  }
    else {
      s->gm_index[s->num_minima + s->num_global_minima - 1 - i] = i;
    }
  }

  if (s->num_global_minima == 0)
    return GKLS_FLOATING_POINT_ERROR;

  return GKLS_OK;
}

gkls_error gkls_internal_check_calc_arguments(state_ptr s, const double *x, double *f) {
  size_t i;
  double tmp1, tmp2;

  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  GKLS_CHECK_CONDITION(s->is_arg_set, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(!s->is_released, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(x != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);
  GKLS_CHECK_CONDITION(f != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);

  for (i = 0; i < s->allocated_dim; i++) {
    tmp1 = s->domain_left[i] - x[i];
    tmp2 = x[i] - s->domain_right[i];
    GKLS_CHECK_CONDITION(tmp1 < GKLS_PRECISION, GKLS_ARGUMENT_OUTSIDE_OF_DOMAIN_ERROR);
    GKLS_CHECK_CONDITION(tmp2 < GKLS_PRECISION, GKLS_ARGUMENT_OUTSIDE_OF_DOMAIN_ERROR);
  }

  return GKLS_OK;
}

gkls_error gkls_create(gkls_generator_state *state) {
  size_t i;
  gkls_error error;
  gkls_generator_state_internal *instate;

  GKLS_CHECK_CONDITION(instate = GKLS_MALLOC(gkls_generator_state_internal, 1), GKLS_MEMORY_ERROR);

  instate->dim = 2;
  instate->num_minima = 10;
  instate->number = 1;
  instate->global_dist = 2.0 / 3.0;
  instate->global_radius = 1.0 / 3.0;
  instate->global_value = GKLS_GLOBAL_MIN_VALUE;
  instate->rnd_counter = 0L;
  instate->num_global_minima = 0;
  instate->is_released = 0;
  instate->is_arg_set = 0;

  instate->domain_left = NULL;
  instate->domain_right = NULL;
  instate->local_min = NULL;
  instate->w_rho = NULL;
  instate->peak = NULL;
  instate->rho = NULL;
  instate->f = NULL;
  instate->rnd_num = NULL;
  instate->ran_u = NULL;
  instate->gm_index = NULL;

  instate->allocated_dim = 0;
  instate->allocated_num_minima = 0;
  instate->allocated_left_domain_dim = 0;
  instate->allocated_right_domain_dim = 0;

  error = gkls_internal_alloc_bounds(instate);
  if (error != GKLS_OK) {
    gkls_free((gkls_generator_state)instate);
    return error;
  }

  for (i = 0; i < instate->dim; i++) {
    instate->domain_left[i]  = -1.0;
    instate->domain_right[i] =  1.0;
  }

  *state = instate;
  return GKLS_OK;
}

gkls_error gkls_copy(gkls_generator_state source, gkls_generator_state *dest) {
  size_t i;
  gkls_error error;
  state_ptr source_s = (state_ptr)source;
  state_ptr s;
  GKLS_CHECK_CONDITION(source_s != NULL, GKLS_NULL_POINTER_STATE_ERROR);

  if (source_s->is_released) {
    return GKLS_NOT_INITIALIZED_ERROR;
  }

  error = gkls_create(dest); GKLS_THROW_ERROR(error);
  s = (state_ptr)(*dest);
  *s = *source_s;

  GKLS_CHECK_CONDITION(s->domain_left =
    GKLS_MALLOC(double, s->allocated_left_domain_dim), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->domain_right =
    GKLS_MALLOC(double, s->allocated_right_domain_dim), GKLS_MEMORY_ERROR);

  memcpy(s->domain_left, source_s->domain_left, sizeof(double) * s->allocated_left_domain_dim);
  memcpy(s->domain_right, source_s->domain_right, sizeof(double) * s->allocated_right_domain_dim);

  if (!source_s->is_arg_set) {
    return GKLS_OK;
  }

  GKLS_CHECK_CONDITION(s->local_min =
    GKLS_MALLOC(double*, s->allocated_num_minima), GKLS_MEMORY_ERROR);
  for (i = 0; i < s->allocated_num_minima; i++) {
    GKLS_CHECK_CONDITION(s->local_min[i] = GKLS_MALLOC(double, s->allocated_dim), GKLS_MEMORY_ERROR);
    memcpy(s->local_min[i], source_s->local_min[i], sizeof(double) * s->allocated_dim);
  }

  GKLS_CHECK_CONDITION(s->w_rho = GKLS_MALLOC(double, s->allocated_num_minima), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->peak = GKLS_MALLOC(double, s->allocated_num_minima), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->rho = GKLS_MALLOC(double, s->allocated_num_minima), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->f = GKLS_MALLOC(double, s->allocated_num_minima), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->rnd_num = GKLS_MALLOC(double, GKLS_RAND_NUM_RND), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->ran_u = GKLS_MALLOC(double, GKLS_RAND_KK), GKLS_MEMORY_ERROR);
  GKLS_CHECK_CONDITION(s->gm_index = GKLS_MALLOC(size_t, s->allocated_num_minima), GKLS_MEMORY_ERROR);

  memcpy(s->w_rho, source_s->w_rho, sizeof(double) * s->allocated_num_minima);
  memcpy(s->peak, source_s->peak, sizeof(double) * s->allocated_num_minima);
  memcpy(s->rho, source_s->rho, sizeof(double) * s->allocated_num_minima);
  memcpy(s->f, source_s->f, sizeof(double) * s->allocated_num_minima);
  memcpy(s->rnd_num, source_s->f, sizeof(double) * GKLS_RAND_NUM_RND);
  memcpy(s->ran_u, source_s->ran_u, sizeof(double) * GKLS_RAND_KK);
  memcpy(s->gm_index, source_s->gm_index, sizeof(double) * s->allocated_num_minima);

  return GKLS_OK;
}

gkls_error gkls_generate(gkls_generator_state state) {
  uint i, j;
  int error;
  double sin_phi, tmp1, tmp2;
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);

  GKLS_CHECK_CONDITION(s->number >= 1 && s->number <= 100, GKLS_FUNC_NUMBER_ERROR);
  GKLS_CHECK_CONDITION(s->dim == s->allocated_left_domain_dim, GKLS_BOUNDARY_ERROR);
  GKLS_CHECK_CONDITION(s->dim == s->allocated_right_domain_dim, GKLS_BOUNDARY_ERROR);

  error = gkls_internal_parameters_check(s); GKLS_THROW_ERROR(error);
  error = gkls_internal_alloc(s);
  if (error != GKLS_OK) {
    gkls_free((gkls_generator_state)s);
    return error;
  }

  gkls_internal_initialize_rnd(s, s->number);
  gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);

  for (i = 0; i < s->num_minima; i++) {
    for (j = 0; j < s->dim; j++) {
      s->local_min[i][j] = 0;
    }
  }

  s->rnd_counter = 0L;
  for (i = 0; i < s->dim; i++) {
    s->local_min[0][i] = s->domain_left[i] +
      s->rnd_num[s->rnd_counter] * (s->domain_right[i] - s->domain_left[i]);
    s->rnd_counter++;
    if (s->rnd_counter == GKLS_RAND_NUM_RND) {
      gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);
      s->rnd_counter = 0L;
    }
  }

  s->f[0] = GKLS_PARABOLOID_MIN;
  gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);

  s->rnd_counter = 0L;
  s->local_min[1][0] = s->local_min[0][0] +
    s->global_dist * cos(GKLS_PI * s->rnd_num[s->rnd_counter]);

  tmp1 = s->domain_right[0] - s->local_min[1][0];
  tmp2 = s->local_min[1][0] - s->domain_left[0];

  if (tmp1 < GKLS_PRECISION || tmp2 < GKLS_PRECISION) {
    s->local_min[1][0] = s->local_min[0][0] - s->global_dist *
      cos(GKLS_PI * s->rnd_num[s->rnd_counter]);
  }

  sin_phi = sin(GKLS_PI * s->rnd_num[s->rnd_counter]);
  s->rnd_counter++;

  for (j = 1; j < s->dim - 1; j++) {
    s->local_min[1][j] = s->local_min[0][j] +
		  s->global_dist * cos(2.0 * GKLS_PI * s->rnd_num[s->rnd_counter]) * sin_phi;

    tmp1 = s->domain_right[j] - s->local_min[1][j];
    tmp2 = s->local_min[1][j] - s->domain_left[j];
    if (tmp1 < GKLS_PRECISION || tmp2 < GKLS_PRECISION) {
      s->local_min[1][j] = s->local_min[0][j] -
        s->global_dist * cos(2.0 * GKLS_PI * s->rnd_num[s->rnd_counter]) * sin_phi;
    }

    sin_phi *= sin(2.0 * GKLS_PI * s->rnd_num[s->rnd_counter]);
    s->rnd_counter++;
  }

  s->local_min[1][s->dim - 1] = s->local_min[0][s->dim - 1] + s->global_dist * sin_phi;

  tmp1 = s->domain_right[s->dim - 1] - s->local_min[1][s->dim - 1];
  tmp2 = s->local_min[1][s->dim - 1] - s->domain_left[s->dim - 1];

  if (tmp1 < GKLS_PRECISION || tmp2 < GKLS_PRECISION) {
    s->local_min[1][s->dim - 1] = s->local_min[0][s->dim - 1] - s->global_dist * sin_phi;
  }

  for (i = 0; i < s->num_minima; i++)
    s->w_rho[i] = 0.99;

  s->w_rho[1] = 1.0;
  s->f[1] = s->global_value;
  s->delta = GKLS_DELTA_MAX_VALUE * s->rnd_num[s->rnd_counter];

  s->rnd_counter++;
  if (s->rnd_counter == GKLS_RAND_NUM_RND) {
    gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);
    s->rnd_counter = 0L;
  }

  do {
    i = 2;
    while (i < s->num_minima) {
      do {
        gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);
        s->rnd_counter = 0L;

        for (j = 0; j < s->dim; j++) {
	        s->local_min[i][j] = s->domain_left[j] +
		        s->rnd_num[s->rnd_counter] * (s->domain_right[j] - s->domain_left[j]);
	        s->rnd_counter++;
	        if (s->rnd_counter == GKLS_RAND_NUM_RND) {
            gkls_internal_ranf_array(s->ran_u, s->rnd_num, GKLS_RAND_NUM_RND);
            s->rnd_counter = 0L;
		      }
	      }

        tmp1 = 2.0 * s->global_radius -
          gkls_internal_norm(s->local_min[i], s->local_min[1], s->dim);
      } while (tmp1 > GKLS_PRECISION);
      i++;
    }
    error = gkls_internal_coincidence_check(s);
  } while (error == GKLS_PARABOLA_MIN_COINCIDENCE_ERROR ||
           error == GKLS_LOCAL_MIN_COINCIDENCE_ERROR);

  error = gkls_internal_set_basins(s);
  if (error == GKLS_OK)
    s->is_arg_set = 1;

  return error;
}

gkls_error gkls_free(gkls_generator_state state) {
  uint i;
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);

  if (s->is_released)
    return GKLS_OK;

  s->is_released = 1;
  s->is_arg_set = 0;

  if (s->is_arg_set) {
    for (i = 0; i < s->allocated_num_minima; i++)
      GKLS_FREE(s->local_min[i]);
  }

  GKLS_FREE(s->local_min);
  GKLS_FREE(s->w_rho);
  GKLS_FREE(s->peak);
  GKLS_FREE(s->rho);
  GKLS_FREE(s->f);
  GKLS_FREE(s->gm_index);
  GKLS_FREE(s->rnd_num);
  GKLS_FREE(s->ran_u);

  s->dim = s->allocated_dim = 0;
  s->num_minima = s->allocated_num_minima = 0;
  s->allocated_left_domain_dim = 0;
  s->allocated_left_domain_dim = 0;

  return GKLS_OK;
}

gkls_error gkls_set_dim(gkls_generator_state state, size_t dim) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->dim = dim;
  return GKLS_OK;
}

gkls_error gkls_set_num_minima(gkls_generator_state state, size_t num) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->num_minima = num;
  return GKLS_OK;
}

gkls_error gkls_set_function_number(gkls_generator_state state, uint number) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->number = number;
  return GKLS_OK;
}

gkls_error gkls_set_global_dist(gkls_generator_state state, double dist) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->global_dist = dist;
  return GKLS_OK;
}

gkls_error gkls_set_global_value(gkls_generator_state state, double value) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->global_value = value;
  return GKLS_OK;
}

gkls_error gkls_set_global_radius(gkls_generator_state state, double radius) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  s->global_radius = radius;
  return GKLS_OK;
}

gkls_error gkls_set_domain_left(gkls_generator_state state, const double *left) {
  gkls_error error;
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  GKLS_CHECK_CONDITION(left != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);

  if (s->dim != s->allocated_left_domain_dim) {
    error = gkls_internal_alloc_left_bound(s); GKLS_THROW_ERROR(error);
  }

  memcpy(s->domain_left, left, s->dim * sizeof(double));
  return GKLS_OK;
}

gkls_error gkls_set_domain_right(gkls_generator_state state, const double *right) {
  gkls_error error;
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  GKLS_CHECK_CONDITION(right != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);

  if (s->dim != s->allocated_right_domain_dim) {
    error = gkls_internal_alloc_right_bound(s); GKLS_THROW_ERROR(error);
  }

  memcpy(s->domain_right, right, s->dim * sizeof(double));
  return GKLS_OK;
}

gkls_error gkls_get_global_minimizer(gkls_generator_state state, size_t i, double *minimizer) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  GKLS_CHECK_CONDITION(s->is_arg_set, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(!s->is_released, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(i < s->num_global_minima, GKLS_INVALID_INDEX_ERROR);
  GKLS_CHECK_CONDITION(i >= 0, GKLS_INVALID_INDEX_ERROR);
  GKLS_CHECK_CONDITION(minimizer != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);
  memcpy(minimizer, s->local_min[s->gm_index[i]], sizeof(double) * s->allocated_dim);
  return GKLS_OK;
}

gkls_error gkls_get_global_minimizers_num(gkls_generator_state state, size_t *num) {
  state_ptr s = (state_ptr)state;
  GKLS_CHECK_CONDITION(s != NULL, GKLS_NULL_POINTER_STATE_ERROR);
  GKLS_CHECK_CONDITION(s->is_arg_set, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(!s->is_released, GKLS_NOT_INITIALIZED_ERROR);
  GKLS_CHECK_CONDITION(num != NULL, GKLS_NULL_POINTER_ARGUMENT_ERROR);
  *num = s->num_global_minima;
  return GKLS_OK;
}

gkls_error gkls_calculate_nd(gkls_generator_state state, const double *x, double *f) {
  size_t i, index;
  gkls_error error;
  double norm, scal, a, rho;
  state_ptr s = (state_ptr)state;

  error = gkls_internal_check_calc_arguments(s, x, f); GKLS_THROW_ERROR(error);

  for (index = 1; index < s->allocated_num_minima; index++) {
    norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);
    if (norm < s->rho[index])
      break;
  }

  if (index == s->allocated_num_minima) {
    norm = gkls_internal_norm(s->local_min[0], x, s->allocated_dim);
    *f = norm * norm + s->f[0];
    return GKLS_OK;
  }

  if (gkls_internal_norm(x, s->local_min[index], s->allocated_dim) < GKLS_PRECISION) {
    *f = s->f[index];
    return GKLS_OK;
  }

  norm = gkls_internal_norm(s->local_min[0], s->local_min[index], s->allocated_dim);
  a = norm * norm + s->f[0] - s->f[index];
  rho = s->rho[index];

  norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);
  scal = 0.0;
  for (i = 0; i < s->allocated_dim; i++) {
    scal += (x[i] - s->local_min[index][i]) *
      (s->local_min[0][i] - s->local_min[index][i]);
  }

  *f = (1.0 - 2.0 / rho * scal / norm + a / rho / rho) * norm * norm + s->f[index];
  return GKLS_OK;
}

gkls_error gkls_calculate_d(gkls_generator_state state, const double *x, double *f) {
  size_t i, index;
  gkls_error error;
  double norm, scal, a, rho;
  state_ptr s = (state_ptr)state;

  error = gkls_internal_check_calc_arguments(s, x, f); GKLS_THROW_ERROR(error);

  for (index = 1; index < s->allocated_num_minima; index++) {
    norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);
    if (norm < s->rho[index])
      break;
  }

  if (index == s->allocated_num_minima) {
    norm = gkls_internal_norm(s->local_min[0], x, s->allocated_dim);
    *f = norm * norm + s->f[0];
    return GKLS_OK;
  }

  if (gkls_internal_norm(x, s->local_min[index], s->allocated_dim) < GKLS_PRECISION) {
    *f = s->f[index];
    return GKLS_OK;
  }

  norm = gkls_internal_norm(s->local_min[0], s->local_min[index], s->allocated_dim);
  a = norm * norm + s->f[0] - s->f[index];
  rho = s->rho[index];
  norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);

  scal = 0.0;
  for(i = 0; i < s->allocated_dim; i++) {
    scal += (x[i] - s->local_min[index][i]) *
      (s->local_min[0][i] - s->local_min[index][i]);
  }

  *f = (2.0 / rho / rho * scal / norm - 2.0 * a / rho / rho /rho) * norm * norm * norm +
       (1.0 - 4.0 * scal / norm/ rho + 3.0 * a / rho / rho) * norm * norm + s->f[index];
  return GKLS_OK;
}


gkls_error gkls_calculate_d2(gkls_generator_state state, const double *x, double *f) {
  size_t i, index;
  gkls_error error;
  double norm, scal, a, rho;
  state_ptr s = (state_ptr)state;

  error = gkls_internal_check_calc_arguments(s, x, f); GKLS_THROW_ERROR(error);

  for (index = 1; index < s->allocated_num_minima; index++) {
    norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);
    if (norm < s->rho[index])
      break;
  }

  if (index == s->allocated_num_minima) {
    norm = gkls_internal_norm(s->local_min[0], x, s->allocated_dim);
    *f = norm * norm + s->f[0];
    return GKLS_OK;
  }

  if (gkls_internal_norm(x, s->local_min[index], s->allocated_dim) < GKLS_PRECISION) {
    *f = s->f[index];
    return GKLS_OK;
  }

  norm = gkls_internal_norm(s->local_min[0], s->local_min[index], s->allocated_dim);
  a = norm * norm + s->f[0] - s->f[index];
  rho = s->rho[index];
  norm = gkls_internal_norm(s->local_min[index], x, s->allocated_dim);

  scal = 0.0;
  for (i = 0; i < s->allocated_dim; i++) {
    scal += (x[i] - s->local_min[index][i]) *
      (s->local_min[0][i] - s->local_min[index][i]);
  }

  *f = ((-6.0 * scal / norm / rho + 6.0 * a / rho / rho + 1.0 - s->delta / 2.0) *
       norm * norm / rho / rho  +
       (16.0 * scal / norm / rho - 15.0 * a / rho / rho - 3.0 + 1.5 * s->delta) * norm / rho +
       (-12.0 * scal /norm / rho + 10.0 * a /rho /rho + 3.0 - 1.5 * s->delta)) *
       norm * norm * norm / rho + 0.5 * s->delta * norm * norm + s->f[index];
  return GKLS_OK;
}

gkls_error gkls_get_error_description(gkls_error code, gkls_error_description *desc) {
  switch (code) {
    case GKLS_OK:
      *desc = "No errors"; return GKLS_OK;
    case GKLS_DIM_ERROR:
      *desc = "Invalid dimension"; return GKLS_OK;
    case GKLS_NUM_MINIMA_ERROR:
      *desc = "Invalid number of minimums"; return GKLS_OK;
    case GKLS_FUNC_NUMBER_ERROR:
      *desc = "Invalid function number"; return GKLS_OK;
    case GKLS_BOUNDARY_ERROR:
      *desc = "Invalid boundaries"; return GKLS_OK;
    case GKLS_GLOBAL_MIN_VALUE_ERROR:
      *desc = "Invalid global minimum"; return GKLS_OK;
    case GKLS_GLOBAL_DIST_ERROR:
      *desc = "Invalid global distance"; return GKLS_OK;
    case GKLS_GLOBAL_RADIUS_ERROR:
      *desc = "Invalid global radius"; return GKLS_OK;
    case GKLS_MEMORY_ERROR:
      *desc = "Can't allocate memory"; return GKLS_OK;
    case GKLS_DERIV_EVAL_ERROR:
      *desc = "Unknown"; return GKLS_OK;
    case GKLS_GREAT_DIM:
      *desc = "Unknown"; return GKLS_OK;
    case GKLS_RHO_ERROR:
      *desc = "Unknown"; return GKLS_OK;
    case GKLS_PEAK_ERROR:
      *desc = "Unknown"; return GKLS_OK;
    case GKLS_GLOBAL_BASIN_INTERSECTION:
      *desc = "Basins of the generated paraboloids are intersected"; return GKLS_OK;
    case GKLS_PARABOLA_MIN_COINCIDENCE_ERROR:
    case GKLS_LOCAL_MIN_COINCIDENCE_ERROR:
      *desc = "Generated local minimums of paraboloids are the same"; return GKLS_OK;
    case GKLS_FLOATING_POINT_ERROR:
      *desc = ""; return GKLS_OK;
    case GKLS_NOT_INITIALIZED_ERROR:
      *desc = "GKLS generator is not initialized"; return GKLS_OK;
    case GKLS_ARGUMENT_OUTSIDE_OF_DOMAIN_ERROR:
      *desc = "Point is outsize of the GKLS domain"; return GKLS_OK;
    case GKLS_NULL_POINTER_STATE_ERROR:
      *desc = "GKLS state is null pointer"; return GKLS_OK;
    case GKLS_NULL_POINTER_ARGUMENT_ERROR:
      *desc = "Argument passed to the function is null pointer"; return GKLS_OK;
    case GKLS_INVALID_INDEX_ERROR:
      *desc = "Index is out of range"; return GKLS_OK;
    case GKLS_INVALID_ERROR_CODE:
      *desc = "Can't find error description for the specified error code"; return GKLS_OK;
  }
  return GKLS_INVALID_ERROR_CODE;
}
