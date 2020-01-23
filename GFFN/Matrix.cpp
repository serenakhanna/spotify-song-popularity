#include <array>
#include <iostream>
#include <math.h>
#include <random>
using std::array;
using std::cout;

template <class T, unsigned int n, unsigned int m>
class Matrix 
{
    private:
        array<array<T,m>,n> mat;
    public:
        const unsigned int rows;
        const unsigned int cols;
        
        Matrix();
        array<T,m>&          operator[]   (int index);
        const array<T,m>&    operator[]   (int index) const;
        Matrix<T,n,m>&       operator=    (const Matrix<T,n,m>& rhs);
        
        void                 show         () const;
        array<T,n>           mvproduct    (array<T, m>) const;
        static Matrix<T,n,m> randomMatrix (int);
        void                 addToRow     (int, float, array<T,m>);
        void                 addToColumn  (int, float, array<T,n>);
        void                 zero         ();
          
};

// --- Definitions ---

template <class T, unsigned int n, unsigned int m>
Matrix<T,n,m>::Matrix() : rows(n), cols(m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mat[i][j] = 0; 
        }
    }
}


template <class T, unsigned int n, unsigned int m>
array<T,m>& Matrix<T,n,m>::operator[] (int index)
{
    return mat[index];
}

template <class T, unsigned int n, unsigned int m>
const array<T,m>& Matrix<T,n,m>::operator[] (int index) const
{
    return mat[index];
}

template <class T, unsigned int n, unsigned int m>
Matrix<T,n,m>& Matrix<T,n,m>::operator=(const Matrix<T,n,m>& rhs)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mat[i][j] = rhs[i][j]; 
        }
    }
}

template <class T, unsigned int n, unsigned int m>
void Matrix<T,n,m>::show() const 
{
    cout << " -";
    for (int k = 0; k < m*(13); k++)
    {
        cout << "-";
    }
    cout << "\n";
    for (int i = 0; i < n; i++)
    {

        cout << " |";
        for (int j = 0; j < m; j++)
        {
            printf(" %10.3e |", (double) mat[i][j]);
        }
        cout << "\n";
    }
    cout << " -";
    for (int k = 0; k < m*(13); k++)
    {
        cout << "-";
    }
    cout << "\n";
}

template <class T, unsigned int n, unsigned int m>
array<T,n> Matrix<T,n,m>::mvproduct(array<T,m> vec) const
{
    T sum;
    array<T, n> product;
    for (int i = 0; i < n; i ++)
    {
        sum = 0;
        for (int j = 0; j < m; j++)
        {
            sum += mat[i][j] * vec[j];
        }
        product[i] = sum;
    }
    return product;
}

template <class T, unsigned int n, unsigned int m>
Matrix<T,n,m> Matrix<T,n,m>::randomMatrix (int seed)
{
    std::default_random_engine e(seed);
    std::uniform_real_distribution<> dist(-1, 1);
    Matrix<T,n,m> newmat;
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
            newmat[i][j] = dist(e) * 0.01;
        }
    }  
    return newmat;
}

template <class T, unsigned int n, unsigned int m>
void Matrix<T,n,m>::addToRow(int index, float rate, array<T,m> row) 
{
    for (int i = 0; i < m; i++)
    {
        mat[index][i] += row[i]*rate;
    }
}

template <class T, unsigned int n, unsigned int m>
void Matrix<T,n,m>::addToColumn(int index, float rate, array<T,n> col) 
{
    for (int i = 0; i < n; i++)
    {
        mat[i][index] += col[i]*rate;
    }
}

template <class T, unsigned int n, unsigned int m>
void Matrix<T,n,m>::zero() {
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
            mat[i][j] = 0;
        }
    }  

}
