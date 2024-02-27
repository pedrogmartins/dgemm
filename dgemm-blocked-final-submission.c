const char* dgemm_desc = "Simple blocked dgemm.";
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define n_c 24 //This parameter can be changed
#define k_c 24 //This parameter can be changed
#define m_c 24 //This parameter can be changed
#define n_r 8 //This parameter can be changed
#define m_r 8 //This parameter is hardcoded on microkernel size
#define k_r 8 //This parameter is used for further blocking

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int k, int n, double* A, double* B, double* C) {


    //Loop over the n_r columns of C block and n_r elements of a B row
    for (int o = 0; o < n;  o++) {

        __m256d C_col_1, C_col_2;

        // Load the related columns of C to populate
        C_col_1 = _mm256_loadu_pd(&C[lda*o]);
        C_col_2 = _mm256_loadu_pd(&C[lda*o + 4]);

        //Loop over the k_c rows of B
        for (int m= 0; m < k; m++) {

            // Declare variables
            __m256d A_col_1, A_col_2;
            __m256d B_el;

            // First we need to get the appropriate columns of A
            A_col_1 = _mm256_loadu_pd(&A[lda*m]);
            A_col_2 = _mm256_loadu_pd(&A[lda*m + 4]);

            // Now we need the matrix B entry
            B_el = _mm256_set1_pd(B[lda*o + m]);

            // Perform FMA operation
            C_col_1 = _mm256_fmadd_pd(A_col_1, B_el, C_col_1);
            C_col_2 = _mm256_fmadd_pd(A_col_2, B_el, C_col_2);

        }

        // Write information back in column of C
        _mm256_storeu_pd(&C[lda*o],  C_col_1);
        _mm256_storeu_pd(&C[lda*o + 4],  C_col_2);

    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int lda, double* A, double* B, double* C) {

    //First, we have to implement padding to make sure the blocking will be executed properly
    int excess = lda%n_c;
    int total = lda;
    if (excess != 0) {
        total += n_c - excess;
    }
    if (n_c > lda) {
        total = n_c;
    }

    double *A_p = (double *) _mm_malloc(total * total * sizeof(double), 64);
    double *B_p = (double *) _mm_malloc(total * total * sizeof(double), 64);
    double *C_p = (double *) _mm_malloc(total * total * sizeof(double), 64);
    memset(C_p, 0, total * total * sizeof(double));

    for (int j = 0; j < total; j++) {
        for (int i = 0; i < total; i++) {
            if (i >= lda || j >= lda) {
                A_p[i + j * total] = 0;
                B_p[i + j * total] = 0;
            } else {
                A_p[i + j * total] = A[i + j * lda];
                B_p[i + j * total] = B[i + j * lda];
            }
        }
    }


    // For each block-row of A
    for (int j = 0; j < total; j += n_c) {
        // For each block-column of B
        for (int k = 0; k < total; k += k_c) {
            // Accumulate block dgemms into block of C
            for (int i = 0; i < total; i += m_c) {

                int A_bord = min(m_c, total - i);
                int B_bord = min(n_c, total - j);
                int C_bord = min(k_c, lda - k);

                double *A_b = A_p + i + k * total;
                double *B_b = B_p + k + j * total;
                double *C_b = C_p + i + j * total;

                for (int b = 0; b < B_bord; b += n_r) {
                    for (int c = 0; c < C_bord; c += m_r) {
                       for(int a = 0; a < A_bord; a +=k_r) {

                           double *A_b_2 = A_b + a + c * total;;
                           double *B_b_2 = B_b + c + b * total;
                           double *C_b_2 = C_b + a + b * total;

                           do_block(total, k_r, n_r, A_b_2, B_b_2, C_b_2);
	 	      }
                   }
                }
            }
        }
    }


   for (int j  = 0; j < lda; j++) {
        for (int i = 0; i < lda; i++) {
           C[i + j * lda] = C_p[i + j * total];
        }
   }

   free(A_p);
   free(B_p);
   free(C_p);
}
