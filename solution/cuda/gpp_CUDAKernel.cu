#include "CustomComplex.h"
#include "commonDefines.h"

__global__ void noflagOCC_cudaSolver(
    int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv,
    dataType *wx_array, CustomComplex<dataType> *wtilde_array,
    CustomComplex<dataType> *aqsmtemp, CustomComplex<dataType> *aqsntemp,
    CustomComplex<dataType> *I_eps_array, dataType *vcoul, dataType *achtemp_re,
    dataType *achtemp_im) {
  dataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
  for (int iw = nstart; iw < nend; ++iw) {
    achtemp_re_loc[iw] = 0.00;
    achtemp_im_loc[iw] = 0.00;
  }

  for (int n1 = blockIdx.x; n1 < number_bands;
       n1 += gridDim.x) // 512 iterations
  {
    for (int my_igp = blockIdx.y; my_igp < ngpown;
         my_igp += gridDim.y) // 1634 iterations
    {
      int indigp = inv_igp_index[my_igp];
      int igp = indinv[indigp];
      CustomComplex<dataType> sch_store1 =
          CustomComplex_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 *
          vcoul[igp];

      for (int ig = threadIdx.x; ig < ncouls; ig += blockDim.x) {
        // hint: pragma
        for (int iw = nstart; iw < nend; ++iw) // 3 iterations
        {
          CustomComplex<dataType> wdiff =
              wx_array[iw] - wtilde_array(my_igp, ig);
          CustomComplex<dataType> delw =
              wtilde_array(my_igp, ig) * CustomComplex_conj(wdiff) *
              (1 / CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
          CustomComplex<dataType> sch_array =
              delw * I_eps_array(my_igp, ig) * sch_store1;

          achtemp_re_loc[iw] += CustomComplex_real(sch_array);
          achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
        }
      }
    } // ngpown
  }   // number_bands

  // Add the final results here;
  for (int iw = nstart; iw < nend; ++iw) {
    atomicAdd(&achtemp_re[iw], achtemp_re_loc[iw]);
    atomicAdd(&achtemp_im[iw], achtemp_im_loc[iw]);
  }
}

void noflagOCC_cuWrapper(int number_bands, int ngpown, int ncouls,
                         int *inv_igp_index, int *indinv, dataType *wx_array,
                         CustomComplex<dataType> *wtilde_array,
                         CustomComplex<dataType> *aqsmtemp,
                         CustomComplex<dataType> *aqsntemp,
                         CustomComplex<dataType> *I_eps_array, dataType *vcoul,
                         dataType *achtemp_re, dataType *achtemp_im) {

  dim3 grid(number_bands, ngpown, 1);
  dim3 threads(32, 1, 1);
  printf("In noflagOCC_cuWrapper launching a cuda Kernel with grid = "
         "(%d,%d,%d), and threads = (%d,%d,%d) \n",
         number_bands, ngpown, 1, 32, 1, 1);

  noflagOCC_cudaSolver<<<grid, threads>>>(
      number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array,
      wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re,
      achtemp_im);

  cudaDeviceSynchronize();
}
