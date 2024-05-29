#include <iostream>
#include <vector>
#include<cstdlib>


class Matrix{
public:
    std::vector<std::vector<float>> data;
    int rows, cols;

    Matrix(int rows, int cols) : rows(rows), cols(cols){
        data.resize(rows, std::vector<float>(cols, 0.0f));    /*!!!!!!!!!!!!!!!!understand this line*/
    }
    void set2zero(){
        for(int i=0; i<rows; ++i){
            for(int ii=0; ii<cols; ++ii){
                data[i][ii]=0;
            }
        }
    }
};
