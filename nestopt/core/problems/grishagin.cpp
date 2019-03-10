#include "nestopt/core/problems/grishagin.hpp"

#include <cmath>
#include <string>
#include <stdexcept>
#include <cstdio>

namespace nestopt {
namespace core {
namespace problems {
namespace grishagin {

constexpr int NUMBER_MIN = 1;
constexpr int NUMBER_MAX = 100;
constexpr double PI = 3.141592654;

constexpr unsigned char MATCON[10][45] = {
   { 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
     0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0 },
   { 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
     1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1 },
   { 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
     1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0 },
   { 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,
     0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0 },
   { 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
     0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1 },
   { 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0 },
   { 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
     1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
   { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
     1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1 },
   { 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
     1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1 },
   { 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0 },
};

constexpr double MINIMIZERS[] = {
   0.603052, 0.408337,  /*  f(min1)=-13.51436   */
   0.652988, 0.320592,  /*  f(min2)=-11.28447   */
   1.000000, 0.000000,  /*  f(min3)=-13.20907   */
   0.066182, 0.582587,  /*  f(min4)=-11.54117   */
   0.904308, 0.872639,  /*  f(min5)=-9.969261   */
   0.344375, 0.524932,  /*  f(min6)=-9.180137   */
   0.000000, 1.000000,  /*  f(min7)=-9.359486   */
   0.948275, 0.887031,  /*  f(min8)=-11.46999   */
   0.226047, 0.520153,  /*  f(min9)=-11.41470   */
   0.341732, 0.197620,  /*  f(min10)=-12.35783  */
   0.069264, 0.430955,  /*  f(min11)=-8.298100  */
   0.000000, 1.000000,  /*  f(min12)=-10.77891  */
   0.45221 , 0.07292,   /*  f(min13)=-9.639918  */
   0.579769, 0.046396,  /*  f(min14)=-10.84688  */
   0.000000, 1.000000,  /*  f(min15)=-9.998392  */
   0.310179, 1.000000,  /*  f(min16)=-13.27447  */
   0.909758, 0.926195,  /*  f(min17)=-11.20579  */
   0.434562, 0.825608,  /*  f(min18)=-8.555728  */
   0.06686 , 0.77051,   /*  f(min19)=-11.08050  */
   0.641337, 0.135186,  /*  f(min20)=-10.84137  */
   0.885029, 0.390289,  /*  f(min21)=-10.09915  */
   0.649650, 0.414282,  /*  f(min22)=-8.821688  */
   0.142623, 0.157327,  /*  f(min23)=-11.20434  */
   0.862953, 1.000000,  /*  f(min24)=-9.774841  */
   0.46036 , 0.99314,   /*  f(min25)=-9.446216  */
   0.379189, 0.688051,  /*  f(min26)=-9.922234  */
   0.845292, 0.424546,  /*  f(min27)=-9.353030  */
   0.441160, 0.016803,  /*  f(min28)=-8.927842  */
   1.000000, 1.000000,  /*  f(min29)=-11.97038  */
   0.303295, 0.134722,  /*  f(min30)=-11.05922  */
   0.109520, 0.265486,  /*  f(min31)=-8.961329  */
   1.000000, 0.000000,  /*  f(min32)=-10.25347  */
   0.593726, 0.503014,  /*  f(min33)=-13.75610  */
   0.694905, 1.000000,  /*  f(min34)=-10.07243  */
   0.051975, 0.409344,  /*  f(min35)=-10.66758  */
   0.125664, 0.518969,  /*  f(min36)=-9.606340  */
   0.000000, 0.000000,  /*  f(min37)=-11.10867  */
   0.155081, 0.238663,  /*  f(min38)=-8.586483  */
   0.53707 , 0.46181,   /*  f(min39)=-8.823492  */
   0.110985, 0.917791,  /*  f(min40)=-10.22533  */
   1.000000, 0.000000,  /*  f(min41)=-13.84155  */
   0.776095, 0.764724,  /*  f(min42)=-10.76901  */
   0.087367, 0.677632,  /*  f(min43)=-8.574448  */
   0.308037, 0.536113,  /*  f(min44)=-10.40137  */
   0.042100, 0.563607,  /*  f(min45)=-8.889051  */
   0.287025, 0.159219,  /*  f(min46)=-10.44960  */
   0.451926, 0.169839,  /*  f(min47)=-10.45448  */
   0.884761, 0.245341,  /*  f(min48)=-9.749494  */
   0.047782, 0.171633,  /*  f(min49)=-10.80496  */
   0.00000 , 0.41596,   /*  f(min50)=-12.16739  */
   0.192108, 0.303789,  /*  f(min51)=-11.14192  */
   0.554153, 0.809821,  /*  f(min52)=-10.06221  */
   0.91475 , 0.54149,   /*  f(min53)=-9.518639  */
   0.661592, 0.925902,  /*  f(min54)=-9.404589  */
   0.962924, 0.436680,  /*  f(min55)=-10.19342  */
   0.000000, 0.000000,  /*  f(min56)=-9.009641  */
   0.616058, 0.560244,  /*  f(min57)=-10.02504  */
   0.439890, 0.343722,  /*  f(min58)=-10.95753  */
   0.218146, 0.677192,  /*  f(min59)=-9.850361  */
   1.000000, 1.000000,  /*  f(min60)=-11.74782  */
   0.198145, 0.317876,  /*  f(min61)=-10.73907  */
   0.875874, 0.653336,  /*  f(min62)=-9.060382  */
   0.22999 , 0.33624,   /*  f(min63)=-10.74852  */
   0.169351, 0.015656,  /*  f(min64)=-8.455091  */
   0.760073, 0.906035,  /*  f(min65)=-11.29555  */
   0.702941, 0.308403,  /*  f(min66)=-10.47617  */
   0.365371, 0.282325,  /*  f(min67)=-9.285640  */
   0.314012, 0.651377,  /*  f(min68)=-9.592745  */
   0.237687, 0.374368,  /*  f(min69)=-10.82136  */
   0.586334, 0.508672,  /*  f(min70)=-9.353659  */
   0.000000, 0.000000,  /*  f(min71)=-9.803835  */
   0.383319, 1.000000,  /*  f(min72)=-10.86034  */
   0.780103, 0.103783,  /*  f(min73)=-10.98085  */
   0.350265, 0.566946,  /*  f(min74)=-8.966131  */
   0.798535, 0.478706,  /*  f(min75)=-11.00382  */
   0.31759 , 0.06967,   /*  f(min76)=-9.959365  */
   0.715929, 0.704778,  /*  f(min77)=-11.38632  */
   0.563040, 0.442557,  /*  f(min78)=-11.44656  */
   0.565078, 0.322618,  /*  f(min79)=-14.17049  */
   0.146731, 0.510509,  /*  f(min80)=-10.61985  */
   0.000000, 0.543167,  /*  f(min81)=-11.31970  */
   0.208533, 0.454252,  /*  f(min82)=-10.95230  */
   0.155111, 0.972329,  /*  f(min83)=-11.41650  */
   0.000000, 1.000000,  /*  f(min84)=-12.05415  */
   0.336467, 0.909056,  /*  f(min85)=-10.45893  */
   0.57001 , 0.90847,   /*  f(min86)=-9.429290  */
   0.296290, 0.540579,  /*  f(min87)=-10.45261  */
   0.172262, 0.332732,  /*  f(min88)=-10.75306  */
   0.000000, 1.000000,  /*  f(min89)=-9.816527  */
   1.000000, 0.000000,  /*  f(min90)=-12.20688  */
   1.000000, 1.000000,  /*  f(min91)=-10.39147  */
   0.674061, 0.869954,  /*  f(min92)=-9.689030  */
   1.000000, 1.000000,  /*  f(min93)=-11.91809  */
   0.852506, 0.637278,  /*  f(min94)=-10.42941  */
   0.877491, 0.399780,  /*  f(min95)=-10.34945  */
   0.835605, 0.751888,  /*  f(min96)=-9.013732  */
   0.673378, 0.827427,  /*  f(min97)=-8.916823  */
   0.831754, 0.367117,  /*  f(min98)=-8.747803  */
   0.601971, 0.734465,  /*  f(min99)=-12.76981  */
   0.000000, 0.000000   /*  f(min100)=-11.43793 */
};

static void generate(unsigned char k[],
                     unsigned char k1[],
                     int kap1, int kap2) {
   int jct = 0;
   for (int i = kap2; i >= kap1; i--) {
      int tmpJ = k[i] + k1[i] + jct;
      jct = tmpJ / 2;
      k[i] = tmpJ - 2 * jct;
   }

   if (jct != 0) {
      for (int i = kap2; i >= kap1; i--) {
         int tmpJ = k[i] + jct;
         jct = tmpJ / 2;
         k[i] = tmpJ - jct * 2;
      }
   }
}

static Scalar random_20(unsigned char k[]) {
   unsigned char k1[45];
   for (int i = 0; i < 38; i++)
      k1[i] = k[i + 7];

   for (int i = 38; i < 45; i++)
      k1[i] = 0;

   for (int i = 0; i < 45; i++)
      k[i] = std::abs(k[i] - k1[i]);

   for (int i = 27; i < 45; i++)
      k1[i] = k[i - 27];

   for (int i = 0; i < 27; i++)
      k1[i] = 0;

   generate(k, k1, 9, 44);
   generate(k, k1, 0, 8);

   Scalar rndm = 0;
   Scalar de2 = 1;
   for (int i = 0; i < 36; i++) {
      de2 /= 2;
      rndm += k[i + 9] * de2;
   }

   return rndm;
}

static void randomize(int nf, int nc,
                      unsigned char *icnf,
                      Scalar *af, Scalar *bf,
                      Scalar *cf, Scalar *df) {
   int i1, i2, i3;
   constexpr int lst = 10;
   i1 = (nf - 1) / lst;
   i2 = i1 * lst;

   for (int j = 0; j < 45; j++)
      icnf[j] = MATCON[i1][j];

   if (i2 != (nf - 1)) {
      i3 = nf - 1 - i2;
      for (int j = 1; j <= i3; j++) {
         for (int i = 0; i < 196; i++)
            random_20(icnf);
      }
   }

   for (int i = 0; i < nc; i++) {
      for (int j = 0; j < nc; j++) {
         af[j * nc + i] = 2 * random_20(icnf) - 1;
         cf[j * nc + i] = 2 * random_20(icnf) - 1;
      }
   }

   for (int i = 0; i < nc; i++) {
      for (int j = 0; j < nc; j++) {
         bf[j * nc + i] = 2 * random_20(icnf) - 1;
         df[j * nc + i] = 2 * random_20(icnf) - 1;
      }
   }
}

} // namespace grishagin

Grishagin::Grishagin(int number) : number_(number) {
   if (number < grishagin::NUMBER_MIN ||
       number > grishagin::NUMBER_MAX) {
      throw std::invalid_argument("Grishagin's function number must be between " +
                                  std::to_string(grishagin::NUMBER_MIN) + " and " +
                                  std::to_string(grishagin::NUMBER_MAX));
   }

   grishagin::randomize(number_, COEFFICIENTS_DIM,
                        icnf_, af_, bf_, cf_, df_);
}

Scalar Grishagin::Compute(const Vector &x) const {
   if (x.size() != 2) {
      throw std::invalid_argument("Size of the input vector must be 2");
   }

   const Scalar dx = grishagin::PI * x[0];
   const Scalar dy = grishagin::PI * x[1];

   const Scalar sdx = std::sin(dx);
   const Scalar cdx = std::cos(dx);
   const Scalar sdy = std::sin(dy);
   const Scalar cdy = std::cos(dy);

   Scalar snx[COEFFICIENTS_DIM];
   Scalar csx[COEFFICIENTS_DIM];
   Scalar sny[COEFFICIENTS_DIM];
   Scalar csy[COEFFICIENTS_DIM];

   snx[0] = sdx;
   csx[0] = cdx;
   sny[0] = sdy;
   csy[0] = cdy;

   for (int i = 0; i < COEFFICIENTS_DIM - 1; i++) {
      snx[i + 1] = snx[i] * cdx + csx[i] * sdx;
      csx[i + 1] = csx[i] * cdx - snx[i] * sdx;
      sny[i + 1] = sny[i] * cdy + csy[i] * sdy;
      csy[i + 1] = csy[i] * cdy - sny[i] * sdy;
   }

   Scalar sum_x = 0;
   Scalar sum_y = 0;
   for (int i = 0; i < COEFFICIENTS_DIM; i++) {
      const Scalar csx_i = csx[i];
      const Scalar snx_i = snx[i];
      for (int j = 0; j < COEFFICIENTS_DIM; j++) {
         const Scalar snx_i_sny_i = snx_i * sny[j];
         const Scalar csx_i_csy_j = csx_i * csy[j];
         sum_x += af_[i * COEFFICIENTS_DIM + j] * snx_i_sny_i +
                  bf_[i * COEFFICIENTS_DIM + j] * csx_i_csy_j;
         sum_y += cf_[i * COEFFICIENTS_DIM + j] * snx_i_sny_i -
                  df_[i * COEFFICIENTS_DIM + j] * csx_i_csy_j;
      }
   }

   return -std::sqrt(sum_x * sum_x + sum_y * sum_y);
}

Scalar Grishagin::Minimum() const {
   return Compute(Minimizer());
}

Vector Grishagin::Minimizer() const {
   const int idx = 2 * (number_ - 1);
   const Scalar x = grishagin::MINIMIZERS[idx];
   const Scalar y = grishagin::MINIMIZERS[idx + 1];
   return Vector::Copy({ x, y });
}

} // namespace problems
} // namespace core
} // namespace nestopt
