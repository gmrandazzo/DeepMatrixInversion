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
	size_t n, msize, n_max;
	double rmin = -1;
	double rmax = 1;
  int det;
  char smx[25], dmx[25], tmx[25];
	matrix *m, *m_inv;
	if(argc != 3){
		printf("\nUsage: %s [matrix size] [number of samples]\n", argv[0]);
	}
	else{
		msize = atoi(argv[1]);
		n_max = atoi(argv[2]);
    sprintf(smx, "singular_matrix_%lux%lu.mx", msize, msize);
    sprintf(dmx, "dataset_matrix_%lux%lu.mx", msize, msize);
    sprintf(tmx, "target_matrix_%lux%lu.mx", msize, msize);
		//srand(14159265358979323846264338327950288419716939937510);
    srand(curr_milliseconds());
		/*Generate the training/validation dataset */
    n = 0;
    while(n < n_max){
      NewMatrix(&m, msize, msize);
      MatrixRandomFill(m, rmin, rmax);
      det = (int)floor(fabs(MatrixDeterminant(m)));
			if(det == 0){
        /* Ops.. a singular matrix was generated... */
        printf("Ops... this matrix is singular!! -> ");
        printf("Determinant == %d\n", det);
				SaveMatrixToFile(m, smx);
			}
			else{
				initMatrix(&m_inv);
				MatrixInversion(m, &m_inv);
				SaveMatrixToFile(m, dmx);
				SaveMatrixToFile(m_inv, tmx);
				DelMatrix(&m_inv);
        n++;
			}
			DelMatrix(&m);
    }

		/*
		 * Generate the singular matrix testset.
		 * For those matrix determinant must be 0 and
		 * thus no inverse matrix is possible.
		 */
    n = 0;
    while(n < n_max){
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
        n++;
			}
			else{
        /* Ops... this matrix is not singular!! */
				printf("Ops... this matrix is not singular!! -> ");
        printf("Determinant = %d\n", det);
        initMatrix(&m_inv);
        MatrixInversion(m, &m_inv);
        SaveMatrixToFile(m, dmx);
        SaveMatrixToFile(m_inv, tmx);
        DelMatrix(&m_inv);
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
