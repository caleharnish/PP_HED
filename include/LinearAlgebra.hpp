#pragma once

#include <vector>   //for containers
#include <iostream> //for defining output
#include <iomanip>  //for setw
#include <cmath>    //for sqrt and cbrt
#include <complex>  //for complex numbers

template <class ValueType>
class Vector {
public:
  using T = ValueType;

  Vector() {
    iSize_ = 1;
    data_ = std::vector<T>(1);
  }

  Vector(int M) {
    iSize_ = M;
    data_ = std::vector<T>(M);
  }

  const T& operator() (int i) const { return data_[i]; }
  T& operator() (int i) { return data_[i]; }

  int getSize_i() const { return iSize_; }

  //Add and update a vector
  template <class T>
  void operator +=(const Vector<T>& A) {
    auto A_iSize = A.getSize_i();
    assert(iSize_ == A_iSize);

    for (int i = 0; i < A_iSize; ++i) {
      data_[i] += A(i);
    }
  }

  //Subtract and update a vector
  template <class T>
  void operator -=(const Vector<T>& A) {
    auto A_iSize = A.getSize_i();
    assert(iSize_ == A_iSize);

    for (int i = 0; i < A_iSize; ++i) {
      data_[i] -= A(i);
    }
  }

  //Unary minus for a vector, returning a new vector
  template <class T>
  Vector<T> operator -() const {
    Vector<T> out = data_;
    for (int i = 0; i < iSize_; ++i) {
      out(i) *= -1;
    }
    return out;
  }

private:
  int iSize_;
  std::vector<T> data_;
};

//print a vector to console
template<class T>
std::ostream& operator <<(std::ostream& out, const Vector<T>& A) {
  unsigned entryWidth = 4;

  auto A_iSize = A.getSize_i();

  out << "\n";
  for (int i = 0; i < A_iSize; ++i) {
    if (i == 0) {
      out << "⌈" << std::left << std::setw(entryWidth) << A(0) << "⌉\n";
    }
    else if (i == A_iSize - 1){
      out << "⌊" << std::left << std::setw(entryWidth) << A(i) << "⌋\n";
    }
    else out << "|" << std::left << std::setw(entryWidth) << A(i) << "|\n";
  }
  return out;
}

//set all entries in a vector to zero without changing shape
template <class T>
void reset(Vector<T>& A) {
  auto A_iSize = A.getSize_i();

  for (int i = 0; i < A_iSize; ++i) {
    A(i) = T(0);
  }
}

//Add two vectors, returning a new vector
template <class T>
Vector<T> operator +(const Vector<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);

  Vector<T> C(A_iSize);
  for (int i = 0; i < A_iSize; ++i) {
    C(i) = A(i) + B(i);
  }

  return C;
}

//Subtract two vectors, returning a new vector
template <class T>
Vector<T> operator -(const Vector<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);

  Vector<T> C(A_iSize);
  for (int i = 0; i < A_iSize; ++i) {
    C(i) = A(i) - B(i);
  }

  return C;
}

//Dot product of two vectors, returning a scalar
template <class T>
T operator *(const Vector<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);

  T C = 0;
  for (int i = 0; i < A_iSize; ++i) {
    C += A(i) * B(i);
  }

  return C;
}

//Cross product of two vectors, returning a vector
//NOTE: maybe this should be ^ instead? like the wedge product
template <class T>
Vector<T> operator %(const Vector<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  assert(A_iSize == 3);

  Vector<T> C(A_iSize);

  C(0) = A(1) * B(2) - A(2) * B(1);
  C(1) = A(2) * B(0) - A(0) * B(2);
  C(2) = A(0) * B(1) - A(1) * B(0);

  return C;
}

//Scalar product of a vector and a scalar, returning a vector
template <class T>
Vector<T> operator *(const Vector<T>& A, const T scalar) {
  auto A_iSize = A.getSize_i();

  Vector<T> out(A_iSize);
  for (int i = 0; i < A_iSize; ++i) {
    out(i) = A(i) * scalar;
  }

  return out;
}

//Scalar product of a scalar and a vector, returning a vector
template <class T>
Vector<T> operator *(const T scalar, const Vector<T>& A) {
  auto A_iSize = A.getSize_i();

  Vector<T> out(A_iSize);
  for (int i = 0; i < A_iSize; ++i) {
    out(i) = A(i) * scalar;
  }

  return out;
}

//Scalar division of a vector and a scalar, returning a vector
template <class T>
Vector<T> operator /(const Vector<T>& A, const T scalar) {
  auto A_iSize = A.getSize_i();

  Vector<T> out(A_iSize);
  for (int i = 0; i < A_iSize; ++i) {
    out(i) = A(i) / scalar;
  }

  return out;
}

//Frobenius norm of a vector, returning a scalar
template <class T>
T norm(const Vector<T>& A) {
  T dot = A * A;
  return sqrt(dot);
}

template <class ValueType>
class Matrix {
public:
  using T = ValueType;

  Matrix() {
    iSize_ = jSize_ = 1;
    data_ = std::vector<T>(1);
  }

  Matrix(int M, int N) {
    iSize_ = M; jSize_ = N;
    data_ = std::vector<T>(M * N);
  }

  const T& operator() (int i, int j) const { return data_[i * jSize_ + j]; }
  T& operator() (int i, int j) { return data_[i * jSize_ + j]; }

  int getSize_i() const { return iSize_; }
  int getSize_j() const { return jSize_; }

  //Add and update a matrix
  template <class T>
  void operator +=(const Matrix<T>& A) {
    auto A_iSize = A.getSize_i();
    auto A_jSize = A.getSize_j();
    assert(iSize_ == A_iSize);
    assert(jSize_ == A_jSize);

    for (int i = 0; i < A_iSize; ++i) {
      for (int j = 0; j < A_jSize; ++j) {
        data_[i * jSize_ + j] += A(i, j);
      }
    }
  }

  //Subtract and update a matrix
  template <class T>
  void operator -=(const Matrix<T>& A) {
    auto A_iSize = A.getSize_i();
    auto A_jSize = A.getSize_j();
    assert(iSize_ == A_iSize);
    assert(jSize_ == A_jSize);

    for (int i = 0; i < A_iSize; ++i) {
      for (int j = 0; j < A_jSize; ++j) {
        data_[i * jSize_ + j] -= A(i, j);
      }
    }
  }

  //Unary minus for a matrix, returning a new matrix
  template <class T>
  Matrix<T> operator -() const {
    Matrix<T> out = data_;
    for (int i = 0; i < iSize_; ++i) {
      for (int j = 0; j < jSize_; ++j) {
        out(i, j) *= -1;
      }
    }
    return out;
  }

private:
  int iSize_, jSize_;
  std::vector<T> data_;
};

//print a matrix to console
template<class T>
std::ostream& operator <<(std::ostream& out, const Matrix<T>& A) {
  unsigned entryWidth = 5;
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  out << "\n";
  for (int i = 0; i < A_iSize; ++i) {
    if (i == 0) {
      out << "⌈";
      for (int j = 0; j < A_jSize; ++j) {
        out << std::left << std::setw(entryWidth) << A(i, j);
        if (j < A_jSize - 1) out << ", ";
      }
      out << "⌉\n";
    }
    else if (i == A_iSize - 1) {
      out << "⌊";
      for (int j = 0; j < A_jSize; ++j) {
        out << std::left << std::setw(entryWidth) << A(i, j);
        if (j < A_jSize - 1) out << ", ";
      }
      out << "⌋\n";
    }
    else {
      out << "|";
      for (int j = 0; j < A_jSize; ++j) {
        out << std::left << std::setw(entryWidth) << A(i, j);
        if (j < A_jSize - 1) out << ", ";
      }
      out << "|\n";
    }
  }
  return out;
}

//set all entries in a matrix to zero without changing shape
template <class T>
void reset(Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      A(i, j) = T(0);
    }
  }
}

//extract a column from the matrix
template <class T>
Vector<T> getColumn(Matrix<T>& A, int j) {
  auto A_iSize = A.getSize_i();

  Vector<T> column(A_iSize);

  for (int i = 0; i < A_iSize; ++i) {
    column(i) = A(i, j);
  }

  return column;
}

//find a non-zero column from the matrix
template <class T>
Vector<T> findNonzeroColumn(Matrix<T>& A, T basicallyZero) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  Vector<T> vec(A_iSize);

  for (int j = 0; j < A_jSize; ++j) {
    auto temp = getColumn(A, j);
    auto magnitude = norm(temp);
    if (magnitude > basicallyZero) {
      vec = temp / magnitude;
      break;
    }
  }

  return vec;
}

//Add two matrices, returning a matrix
template <class T>
Matrix<T> operator +(const Matrix<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  auto A_jSize = A.getSize_j();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_jSize);

  Matrix<T> C(A_iSize, A_jSize);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }

  return C;
}

//Subtract two matrices, returning a matrix
template <class T>
Matrix<T> operator -(const Matrix<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  auto A_jSize = A.getSize_j();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_jSize);

  Matrix<T> C(A_iSize, A_jSize);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }

  return C;
}

//multiply two vectors, returning a matrix (tensor product or outer product)
template <class T>
Matrix<T> operator &(const Vector<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();

  Matrix<T> C(A_iSize, B_iSize);
  reset(C);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < B_iSize; ++j) {
      C(i, j) = A(i) * B(j);
    }
  }

  return C;
}

//scalar multiply a matrix, returning a matrix
template <class T>
Matrix<T> operator *(T scalar, const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  Matrix<T> C(A_iSize, A_jSize);
  reset(C);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      C(i, j) = A(i, j) * scalar;
    }
  }

  return C;
}

//multiply two matrices, returning a matrix
//NOTE: Dr. Lumsdaine's HPC slides show how this can be sped up
// consider handout #4 and #5:
// "Hoisting" or "Unroll & Jam" or "Tiling & Hoisting", etc.
template <class T>
Matrix<T> operator *(const Matrix<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  auto B_iSize = B.getSize_i();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_iSize);

  Matrix<T> C(A_iSize, A_jSize);
  reset(C);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < B_jSize; ++j) {
      for (int k = 0; k < A_jSize; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }

  return C;
}

//full contraction of two matrices, returning a scalar
template <class T>
T operator |(const Matrix<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  auto A_jSize = A.getSize_j();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_jSize);

  T doubleDot = 0;
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      doubleDot += A(i, j) * B(i, j);
    }
  }

  return doubleDot;
}

//Frobenius norm of a matrix, returning a scalar
template <class T>
T norm(const Matrix<T>& A) {
  T dot = A | A;
  return sqrt(dot);
}

//Determinant of a matrix, returning a scalar
//Frobenius norm of a matrix, returning a scalar
template <class T>
T det(const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  assert (A_iSize == A_jSize);

  if (A_iSize == 2) {
    return A(0,0) * A(1,1) - A(0,1) * A(1,0);
  }
  else if (A_iSize == 3) {
    auto temp1 = A(0,0) * ( A(1,1) * A(2,2) - A(1,2) * A(2,1) );
    auto temp2 = A(0,1) * ( A(1,0) * A(2,2) - A(2,0) * A(1,2) );
    auto temp3 = A(0,2) * ( A(1,0) * A(2,1) - A(1,1) * A(2,0) );
    return temp1 - temp2 + temp3;
  }
  else {
    std::cout << "determinants of this size not supported\n";
    return -1;
  }
}

//matrix-vector product, returning a vector
template <class T>
Vector<T> operator *(const Matrix<T>& A, const Vector<T>& B) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  auto B_iSize = B.getSize_i();
  assert(A_jSize == B_iSize);

  Vector<T> C(A_iSize);
  reset(C);
  for (int i = 0; i < A_iSize; ++i) {
    for (int k = 0; k < A_jSize; ++k) {
      C(i) += A(i, k) * B(k);
    }
  }

  return C;
}

//transpose of a matrix, returning a matrix
template <class T>
Matrix<T> transpose(const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  Matrix<T> C(A_iSize, A_jSize);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      C(i, j) = A(j, i);
    }
  }

  return C;
}

//trace of a matrix, returning a scalar
template <class T>
T tr(const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  assert(A_iSize == A_jSize);

  T trace = 0;
  for (int i = 0; i < A_iSize; ++i) {
    trace += A(i, i);
  }

  return trace;
}

//symmetric part of a matrix, returning a matrix
template <class T>
Matrix<T> sym(const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  assert(A_iSize == A_jSize);

  auto B = T(0.5) * (A + transpose(A));

  return B;
}

//skew-symmetric part of a matrix, returning a matrix
template <class T>
Matrix<T> skw(const Matrix<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  assert(A_iSize == A_jSize);

  auto B = T(0.5) * (A - transpose(A));

  return B;
}

//create an identity matrix
//NOTE: I don't know how to template this for any type T
Matrix<double> identityMatrix(int size) {
  Matrix<double> C(size, size);
  reset(C);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) C(i, j) = 1.0;
    }
  }

  return C;
}

//this bit is used exclusively during the eigenvalue determination because the
//eigenvalues should be returned in order of their magnitude (largest first)
//NOTE: there will only ever be either 2 or 3 values to sort
///https://codereview.stackexchange.com/questions/64758/sort-three-input-values-by-order
template<typename T>
void swap_if_greater(T& a, T& b)
{
    if (std::abs(a) > std::abs(b))
    {
        T tmp(a);
        a = b;
        b = tmp;
    }
}

template<typename T>
void sort(T& a, T& b, T& c)
{
    swap_if_greater(a, b);
    swap_if_greater(a, c);
    swap_if_greater(b, c);
}

//find the eigenvalues for the matrix A
///https://en.wikipedia.org/wiki/Eigenvalue_algorithm
template<class T>
Vector<T> eigenvalues(Matrix<T> A) {
  using std::abs;

  T basicallyZero = 1e-12;

  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  assert(A_iSize == A_jSize); //only for square matrices

  Vector<T> values(A_iSize);

  if(A_iSize == 2) {
    //the roots of the characteristic polynomial:
    //  lambda^2 - tr(A) * lambda + det(A)
    auto trace = tr(A);
    auto determinant = det(A);
    auto gap = sqrt(trace*trace - 4.0*determinant);
    T lambda1 = (trace + gap) * 0.5;
    T lambda2 = (trace - gap) * 0.5;
    //return the values in order of their magnitude (largest first)
    swap_if_greater(lambda1, lambda2);
    values(0) = lambda2;
    values(1) = lambda1;
  }
  else if (A_iSize == 3) {
    //the roots of the characteristic polynomial:
    //  lambda^3 - tr(A) * lambda^2 - 0.5*(tr(A^2) - tr(A)^2) * lambda - det(A)
    auto a = 1.0;
    auto b = - tr(A);
    auto c = - 0.5 * (tr(A * A) - b * b);
    auto d = - det(A);

    //the discriminant of the characteristic polynomial
    auto discriminant = 18.0*a*b*c*d - 4.0*b*b*b*d;
    discriminant += b*b*c*c - 4.0*a*c*c*c - 27.0*a*a*d*d;

    //the types of the eigenvalues are based on the discriminant
    // if discriminant > 0 then there are 3 real eigenvalues
    // if discriminant = 0 then there are 3 real eigenvalues and some are multiple
    // if discriminant < 0 then there is 1 real eigenvalue and 2 complex ones

    //the discriminant of the derivative of the characteristic polynomial
    auto delta0 = b*b - 3.0*a*c;

    if (abs(discriminant) < basicallyZero and
        abs(delta0)       < basicallyZero) {
      //then there is a single eigenvalue
      auto lambda = -b / (3.0 * a);
      values(0) = lambda;
      values(1) = lambda;
      values(2) = lambda;
    }
    else if (abs(discriminant) < basicallyZero and
             abs(delta0)       > basicallyZero) {
      //then there is one unique eigenvalue
      T lambda1 = (4.0*a*b*c - 9.0*a*a*d - b*b*b) / (a * delta0);
      //and there is a duplicate eigenvalue
      T lambda2 = (9.0*a*d - b*c) / (2.0 * delta0);
      T lambda3 = lambda2;
      //return the values in order of their magnitude (largest first)
      sort(lambda1, lambda2, lambda3);
      values(0) = lambda3;
      values(1) = lambda2;
      values(2) = lambda1;
    }
    else {
      //there are three distinct eigenvalues
      std::complex<T> delta1(2.0*b*b*b - 9.0*a*b*c + 27.0*a*a*d, 0.0);
      std::complex<T> delta2(-27.0*a*a*discriminant, 0.0);
      std::complex<T> C0(delta1 + std::sqrt(delta2));

      //C0 must be defined such that it is not zero
      if (abs(C0.real()) < basicallyZero and
          abs(C0.imag()) < basicallyZero) {
        C0 = delta1 - std::sqrt(delta2);
      }
      C0 = 0.5*C0;

      //transform C0 into polar form to do cube root
      auto r     = std::abs(C0);
      auto theta = std::arg(C0);
      std::complex<T> C1 = std::polar(std::cbrt(r), theta / 3.0);
      std::complex<T> C2(-0.5, 0.5*std::sqrt(3.0));
      std::complex<T> C3 = std::conj(C2);
      C2 = C2 * C1;
      C3 = C3 * C1;

      T lambda1 = std::real((b + C1 + delta0 / C1) / (- 3.0 * a));
      T lambda2 = std::real((b + C2 + delta0 / C2) / (- 3.0 * a));
      T lambda3 = std::real((b + C3 + delta0 / C3) / (- 3.0 * a));

      //return the values in order of their magnitude (largest first)
      sort(lambda1, lambda2, lambda3);

      values(0) = lambda3;
      values(1) = lambda2;
      values(2) = lambda1;
    }
  }

  return values;
}

//find the eigenvectors for the matrix A
template<class T>
Matrix<T> eigenvectors(Matrix<T> A, Vector<T> lambdas) {
  using std::abs;

  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  auto NumLambdas = lambdas.getSize_i();

  assert(NumLambdas == A_iSize and NumLambdas == A_jSize);

  T basicallyZero = 1e-12;

  auto lambda1 = lambdas(0);
  auto lambda2 = lambdas(1);
  auto lambda3 = lambdas(2);

  auto matrix1 = (A - lambda2*identityMatrix(3))*(A - lambda3*identityMatrix(3));
  auto matrix2 = (A - lambda1*identityMatrix(3))*(A - lambda3*identityMatrix(3));
  auto matrix3 = (A - lambda1*identityMatrix(3))*(A - lambda2*identityMatrix(3));

  auto vec1 = findNonzeroColumn(matrix1, basicallyZero);
  auto vec2 = findNonzeroColumn(matrix2, basicallyZero);
  auto vec3 = findNonzeroColumn(matrix3, basicallyZero);

  Matrix<T> Q(A_iSize, A_jSize);
  for (int i = 0; i < A_iSize; ++i) {
    Q(i, 0) = vec1(i);
    Q(i, 1) = vec2(i);
    Q(i, 2) = vec3(i);
  }

  return Q;
}

//find the square root for the matrix A, returning a new matrix
template<class T>
Matrix<T> sqrt(Matrix<T> A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  assert(A_iSize == A_jSize);

  auto lambdas = eigenvalues(A);
  auto Q = eigenvectors(A, lambdas);

  auto lambdaMatrix = identityMatrix(3);
  for (int i = 0; i < A_iSize; ++i) {
    lambdaMatrix(i,i) = sqrt(lambdas(i));
  }

  return Q * lambdaMatrix * transpose(Q);
}

//find the inverse for the matrix A, returning a new matrix
template<class T>
Matrix<T> inverse(Matrix<T> A) {

  T basicallyZero = 1e-12;

  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  assert(A_iSize == A_jSize);

  auto lambdas = eigenvalues(A);
  for (int i = 0; i < A_iSize; ++i) {
    assert(lambdas(i) > basicallyZero);
  }
  auto Q = eigenvectors(A, lambdas);

  auto lambdaMatrix = identityMatrix(3);
  for (int i = 0; i < A_iSize; ++i) {
    lambdaMatrix(i,i) = 1.0 / lambdas(i);
  }

  return Q * lambdaMatrix * transpose(Q);
}

template <class ValueType>
class Rank4 {
public:
  using T = ValueType;

  Rank4() {
    iSize_ = jSize_ = 1;
    data_ = std::vector<T>(1);
  }

  Rank4(int M, int N, int O, int P) {
    iSize_ = M; jSize_ = N; kSize_ = O; lSize_ = P;
    data_ = std::vector<T>(M * N * O * P);
  }

  const T& operator() (int i, int j, int k, int l) const {
    auto key = j + i * jSize_ + k * (iSize_ * jSize_) + l * (iSize_ * jSize_ * kSize_);
    return data_[(i * jSize_ + j)];
  }
  T& operator() (int i, int j, int k, int l) {
    auto key = j + i * jSize_ + k * (iSize_ * jSize_) + l * (iSize_ * jSize_ * kSize_);
    return data_[key];
  }

  int getSize_i() const { return iSize_; }
  int getSize_j() const { return jSize_; }
  int getSize_k() const { return kSize_; }
  int getSize_l() const { return lSize_; }

  //Add and update a Rank4
  template <class T>
  void operator +=(const Rank4<T>& A) {
    auto A_iSize = A.getSize_i();
    auto A_jSize = A.getSize_j();
    auto A_kSize = A.getSize_k();
    auto A_lSize = A.getSize_l();

    assert(iSize_ == A_iSize);
    assert(jSize_ == A_jSize);
    assert(kSize_ == A_kSize);
    assert(lSize_ == A_lSize);

    for (int i = 0; i < A_iSize; ++i) {
      for (int j = 0; j < A_jSize; ++j) {
        for (int k = 0; i < A_kSize; ++k) {
          for (int l = 0; j < A_lSize; ++l) {
            auto key = j + i * jSize_ + k * (iSize_ * jSize_) + l * (iSize_ * jSize_ * kSize_);
            data_[key] += A(i, j, k, l);
          }//end l loop
        }//end k loop
      }//end j loop
    }//end i loop
  }

  //Subtract and update a Rank4
  template <class T>
  void operator -=(const Rank4<T>& A) {
    auto A_iSize = A.getSize_i();
    auto A_jSize = A.getSize_j();
    auto A_kSize = A.getSize_k();
    auto A_lSize = A.getSize_l();

    assert(iSize_ == A_iSize);
    assert(jSize_ == A_jSize);
    assert(kSize_ == A_kSize);
    assert(lSize_ == A_lSize);

    for (int i = 0; i < A_iSize; ++i) {
      for (int j = 0; j < A_jSize; ++j) {
        for (int k = 0; i < A_kSize; ++k) {
          for (int l = 0; j < A_lSize; ++l) {
            auto key = j + i * jSize_ + k * (iSize_ * jSize_) + l * (iSize_ * jSize_ * kSize_);
            data_[key] -= A(i, j, k, l);
          }//end l loop
        }//end k loop
      }//end j loop
    }//end i loop
  }

  //Unary minus for a Rank4, returning a new Rank4
  template <class T>
  Rank4<T> operator -() const {
    Rank4<T> out = data_;
    for (int i = 0; i < iSize_; ++i) {
      for (int j = 0; j < jSize_; ++j) {
        for (int k = 0; k < kSize_; ++k) {
          for (int l = 0; l < lSize_; ++l) {
            out(i, j, k, l) *= -1;
          }//end l loop
        }//end k loop
      }//end j loop
    }//end i loop

    return out;
  }

private:
  int iSize_, jSize_, kSize_, lSize_;
  std::vector<T> data_;
};

//set all entries in a Rank4 to zero without changing shape
template <class T>
void reset(Rank4<T>& A) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  auto A_kSize = A.getSize_k();
  auto A_lSize = A.getSize_l();

  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      for (int k = 0; k < A_kSize; ++k) {
        for (int l = 0; l < A_lSize; ++l) {
          A(i, j, k, l) = T(0);
        }
      }
    }
  }
}

//Add two Rank4, returning a new Rank4
template <class T>
Rank4<T> operator +(const Rank4<T>& A, const Rank4<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  auto A_jSize = A.getSize_j();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_jSize);
  auto A_kSize = A.getSize_k();
  auto B_kSize = B.getSize_k();
  assert(A_kSize == B_kSize);
  auto A_lSize = A.getSize_l();
  auto B_lSize = B.getSize_l();
  assert(A_lSize == B_lSize);

  Rank4<T> C(A_iSize, A_jSize, A_kSize, A_lSize);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      for (int k = 0; k < A_kSize; ++k) {
        for (int l = 0; l < A_lSize; ++l) {
          C(i, j, k, l) = A(i, j, k, l) + B(i, j, k, l);
        }
      }
    }
  }

  return C;
}

//Subtract two Rank4, returning a new Rank4
template <class T>
Rank4<T> operator -(const Rank4<T>& A, const Rank4<T>& B) {
  auto A_iSize = A.getSize_i();
  auto B_iSize = B.getSize_i();
  assert(A_iSize == B_iSize);
  auto A_jSize = A.getSize_j();
  auto B_jSize = B.getSize_j();
  assert(A_jSize == B_jSize);
  auto A_kSize = A.getSize_k();
  auto B_kSize = B.getSize_k();
  assert(A_kSize == B_kSize);
  auto A_lSize = A.getSize_l();
  auto B_lSize = B.getSize_l();
  assert(A_lSize == B_lSize);

  Rank4<T> C(A_iSize, A_jSize, A_kSize, A_lSize);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      for (int k = 0; k < A_kSize; ++k) {
        for (int l = 0; l < A_lSize; ++l) {
          C(i, j, k, l) = A(i, j, k, l) - B(i, j, k, l);
        }
      }
    }
  }

  return C;
}

//multiply two matrices, returning a Rank4 (tensor product or outer product)
template <class T>
Rank4<T> operator &(const Matrix<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();
  auto B_iSize = B.getSize_i();
  auto B_jSize = B.getSize_j();

  Rank4<T> C(A_iSize, A_jSize, B_iSize, B_jSize);
  reset(C);
  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      for (int k = 0; k < B_iSize; ++k) {
        for (int l = 0; l < B_jSize; ++l) {
          C(i, j, k, l) = A(i, j) * B(k, l);
        }
      }
    }
  }

  return C;
}

//Rank4-matrix product, returning a matrix (double dot)
template <class T>
Matrix<T> operator |(const Rank4<T>& A, const Matrix<T>& B) {
  auto A_iSize = A.getSize_i();
  auto A_jSize = A.getSize_j();

  auto A_kSize = A.getSize_k();
  auto B_iSize = B.getSize_i();
  assert(A_kSize == B_iSize);
  auto A_lSize = A.getSize_l();
  auto B_jSize = B.getSize_j();
  assert(A_lSize == B_jSize);


  Matrix<T> doubleDot(A_iSize, A_jSize);
  reset(doubleDot);

  for (int i = 0; i < A_iSize; ++i) {
    for (int j = 0; j < A_jSize; ++j) {
      for (int k = 0; k < A_kSize; ++k) {
        for (int l = 0; l < A_lSize; ++l) {
          doubleDot(i, j) += A(i, j, k, l) * B(k, l);
        }
      }
    }
  }

  return doubleDot;
}

//create the first rank 4 identity
//NOTE: I don't know how to template this for any type T
Rank4<double> identity_4(int size) {
  auto delta = identityMatrix(size);

  Rank4<double> C(size, size, size, size);
  reset(C);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        for (int l = 0; l < size; ++l) {
          C(i, j, k, l) *= delta(i, k) * delta(j, l);
        }
      }
    }
  }

  return C;
}

//create the second rank4 identity
//NOTE: I don't know how to template this for any type T
Rank4<double> identity_4_bar(int size) {
  auto delta = identityMatrix(size);

  Rank4<double> C(size, size, size, size);
  reset(C);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        for (int l = 0; l < size; ++l) {
          C(i, j, k, l) *= delta(i, l) * delta(j, k);
        }
      }
    }
  }

  return C;
}
