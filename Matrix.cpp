#include <iostream>
#include <vector>


class Matrix{
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix(int rows, int cols) : rows(rows), cols(cols){
        data.resize(rows, std::vector<double>(cols, 0.0));    /*0.0 initialized the values to zero*/
    }
    void set2zero(){
        for(int i=0; i<rows; ++i){
            for(int ii=0; ii<cols; ++ii){
                data[i][ii]=0;
            }
        }
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(float scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
};

// int main(){
//     return 0;
// }
