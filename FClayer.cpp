// #include "Matrix.cpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include<cstdlib>


// Matrix readPgm(const std::string& filename, const int maxVal){
//     std::ifstream file(filename);
//     if(!file.is_open()){
//         std::cerr << "Unable to open file " << filename << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     std::string line;
//     std::getline(file, line);         /*first line of the file*/
//     if(line!="P2"){
//         std::cerr << "Invalid PGM file " << filename << " - Magic number is not P2" << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     while(std::getline(file, line)){             /*the commented lines of the file*/
//         if(line[0]!='#'){
//             break;
//         }
//     }

//     std::stringstream ss(line);            /*the line afte the commented lines of the file*/
//     int cols, rows;
//     if(!(ss >> rows >> cols)){
//         std::cerr << "Invalid PGM file: " << filename << " - Unable to read image dimensions" << std::endl;
//         exit(EXIT_FAILURE);
//     }
//     // std::cout << rows << "x"<< cols << std::endl;

//     int max_val;
//     file >> max_val;                /*next line containing the max pixel value in the file*/
//     if(max_val!=maxVal){
//         std::cerr << "Invalid PGM file: " << filename << " - Max value is not" << maxVal << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     Matrix img(rows, cols);                         /*NOTE:       try recursive here */
//     for (int i=0; i<rows; ++i){
//         for (int j=0; j<cols; ++j){
//             int pixel;
//             if(!(file >> pixel)){
//                 std::cerr << "Invalid PGM file: " << filename << " - Unable to read pixel value" << std::endl;
//                 exit(EXIT_FAILURE);
//             }
//             img.data[i][j] = static_cast<float>(pixel) / max_val;              /*!!!!!!!!!!!!!!!!what does this line do*/
//         }
//     }

//     file.close();
//     return img;
// }



class FClayer{
public:
    Matrix weights = Matrix(10,512);
    std::vector<float> biases = std::vector<float>(10);

    FClayer(int inputSize, int outSize, const Matrix& inWeights, const Matrix& inBiases) {
        for(int i=0; i<outSize; ++i){
            if(i>4){
                biases[i] = inBiases.data[1][i-5];
            }else{
                biases[i] = inBiases.data[0][i];
            }
            for(int ii=0; ii<inputSize; ++ii){

                // !!!!!Note!!!! recheck this
                weights.data[i][ii] = inWeights.data[i][ii];    /*assigning random numbers to the weights of fullyConnected Layer*/
            }
        }
    }

    std::vector<float> forward(const std::vector<float>& input, std::vector<float>& fc_output){
        for(int i=0; i<biases.size(); ++i){
            fc_output[i] = biases[i];
            for(int j=0; j<input.size(); ++j){

                fc_output[i] += input[j] * weights.data[i][j];
            }
            fc_output[i] = relu(fc_output[i]);
        }
        return fc_output;
    }

    float relu(float x){
        return std::max(0.0f, x);
    }

    void printWeights(){
        for(int i=0; i<10; ++i){
            for(int ii=0; ii<512; ++ii){
                std::cout<<weights.data[i][ii]<<" ";
            }
            std::cout<< "--> " << biases[i]<<std::endl;
        }
    }
};

// int main(){
//     const Matrix inBiases = readPgm("fcB.pmg", 1);
//     const Matrix inWeights = readPgm("fcW.pmg", 1);


//     std::vector<float> img = std::vector<float>(512);
//     std::vector<float> result = std::vector<float>(10);
//     for(int ii=0;ii<512;++ii){
//         img[ii] = 1;
//     }

//     FClayer fc = FClayer(512, 10, inWeights, inBiases);
//     // fc.printWeights();
//     fc.forward(img, result);

//     for(int i=0; i<result.size(); ++i){
//         std::cout << result[i] << " ";
//     }


//     return 0;
// }
