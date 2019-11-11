#include <stdio.h>
#include <stdlib.h>
#include <scientific.h>
#include <math.h>
#include <time.h>

#include <sys/time.h>

size_t curr_milliseconds(){
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    return (size_t) te.tv_usec/1000;
}

double rnorm(x, avg, var)
{
	return 1./sqrt(2*_pi_*var) * exp( -1 * square(x-avg)/(2*var));
}

double randBetween(double min, double max)
{
 return ((double)rand()/RAND_MAX) * (max - min) + min;
}

void MatrixRandomFill(matrix *m, double min, double max)
{
  size_t i, j;
  for(i = 0; i < m->row; i++){
    for(j = 0; j < m->col; j++){
      m->data[i][j] = randBetween(min, max);
    }
  }
}

void ConvertToRNormMatrixValues(matrix *m, double avg, double var)
{
	size_t i, j;
	for(i = 0; i < m->row; i++){
		for(j = 0; j < m->col; j++){
			//double x = randBetween(-2.4, 2.4);
      double x = m->data[i][j];
			m->data[i][j] = rnorm(x, avg, var);
		}
	}
}

void SaveMatrixToFile(matrix *m, char *outfilename)
{
	size_t i, j;
	FILE *fptr;
	fptr = fopen(outfilename, "a");
	for(i = 0; i < m->row; i++){
		for(j = 0; j < m->col-1; j++){
			fprintf(fptr,"%.4f,", m->data[i][j]);
		}
		fprintf(fptr, "%.4f\n", m->data[i][m->col-1]);
	}
	fprintf(fptr, "END\n");
	fclose(fptr);
}

int main(int argc, char **argv){
	size_t n_train, n_val, n_singular, msize, n_max;
	double rmin = atof(argv[3]);
	double rmax = atof(argv[4]);
  int det;
  char smx[35], simx[35], tmx[35], timx[35], vmx[35], vimx[35];
	matrix *m, *m_inv;
	if(argc != 5){
		printf("\nUsage: %s [matrix size] [number of samples] [range min] [range max]\n", argv[0]);
	}
	else{
		msize = atoi(argv[1]);
		n_max = atoi(argv[2]);
    sprintf(smx, "sing_matrix_%lux%lu.mx", msize, msize);
    sprintf(simx, "sing_moore-penrose_inverse_%lux%lu.mx", msize, msize);
    sprintf(tmx, "train_matrix_%lux%lu.mx", msize, msize);
    sprintf(timx, "train_target_inverse_%lux%lu.mx", msize, msize);
    sprintf(vmx, "val_matrix_%lux%lu.mx", msize, msize);
    sprintf(vimx, "val_target_inverse_%lux%lu.mx", msize, msize);
		//srand(14159265358979323846264338327950288419716939937510);
    srand(curr_milliseconds());
    n_train = 0;
    n_singular = 0;
    /*
     * Generate the training set
     */
    while(n_train < n_max){
      NewMatrix(&m, msize, msize);
      MatrixRandomFill(m, rmin, rmax);
      det = (int)floor(fabs(MatrixDeterminant(m)));
			if(det == 0){
        if(n_singular < n_max){
          /* Ops.. a singular matrix was generated... */
          printf("Ops... this matrix is singular!! -> ");
          printf("Determinant == %d\n", det);
  				SaveMatrixToFile(m, smx);
          initMatrix(&m_inv);
          MatrixMoorePenrosePseudoinverse(m, &m_inv);
          SaveMatrixToFile(m_inv, simx);
          DelMatrix(&m_inv);
          n_singular++;
        }
        else{
          continue;
        }
			}
			else{
				initMatrix(&m_inv);
				MatrixInversion(m, &m_inv);
				SaveMatrixToFile(m, tmx);
				SaveMatrixToFile(m_inv, timx);
				DelMatrix(&m_inv);
        n_train++;
			}
			DelMatrix(&m);
    }

    /*
     * Generate the external testing set
     */
    n_val = 0;
    while(n_val < n_max){
      NewMatrix(&m, msize, msize);
      MatrixRandomFill(m, rmin, rmax);
      det = (int)floor(fabs(MatrixDeterminant(m)));
			if(det == 0){
        /* Ops.. a singular matrix was generated... */
        if(n_singular < n_max){
          printf("Ops... this matrix is singular!! -> ");
          printf("Determinant == %d\n", det);
  				SaveMatrixToFile(m, smx);
          initMatrix(&m_inv);
          MatrixMoorePenrosePseudoinverse(m, &m_inv);
          SaveMatrixToFile(m_inv, simx);
          DelMatrix(&m_inv);
          n_singular++;
        }
        else{
          continue;
        }
			}
			else{
				initMatrix(&m_inv);
				MatrixInversion(m, &m_inv);
				SaveMatrixToFile(m, vmx);
				SaveMatrixToFile(m_inv, vimx);
				DelMatrix(&m_inv);
        n_val++;
			}
			DelMatrix(&m);
    }

		/*
		 * Generate the singular matrix testset.
		 * For those matrix determinant must be 0 and
		 * thus no inverse matrix is possible.
		 */
    while(n_singular < n_max){
      /*
      Simplest method to get singular matrix
      A is a n x n square matrix.
      If two rows are = then there the matrix is singular..
      */
      NewMatrix(&m, msize, msize);
      MatrixRandomFill(m, rmin, rmax);
      int r1_row = (int)randBetween(0.f, (double)m->row);
      int r2_row = (int)randBetween(0.f, (double)m->row);
      //printf("copy row %d into %d\n", r2_row, r1_row);
      for(int j = 0; j < m->col; j++){
        m->data[r1_row][j] = m->data[r2_row][j];
      }

      int det = (int)floor(fabs(MatrixDeterminant(m)));
      if(det == 0){
				SaveMatrixToFile(m, smx);
        initMatrix(&m_inv);
        MatrixMoorePenrosePseudoinverse(m, &m_inv);
        SaveMatrixToFile(m_inv, simx);
        DelMatrix(&m_inv);
        n_singular++;
			}
			else{
        continue;
			}
      DelMatrix(&m);

      /*
      More complex method to get singular matrix
      A is a n x m matrix
      singular matrix = A * A_T

      NewMatrix(&m, 3*i, i);
			NewMatrix(&m_t, i, 3*i);
			NewMatrix(&m_m_t, i, i);
      MatrixRandomFill(m, rmin, rmax);

			MatrixTranspose(m, m_t);
			MatrixDotProduct(m_t, m, m_m_t);
			double det = MatrixDeterminant(m_m_t);
			if(FLOAT_EQ(det, 0.f, 1e-1)){
				SaveMatrixToFile(m_m_t, "singular_matrix.mx");
			}
			else{
				printf("Determinant != 0 -> %.4f\n", det);
				PrintMatrix(m_m_t);
			}
			DelMatrix(&m);
			DelMatrix(&m_t);
			DelMatrix(&m_m_t);
      */
		}
	}
	return 0;
}
