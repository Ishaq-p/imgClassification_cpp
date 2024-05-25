#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include<cstdlib>


class Matrix{
public:
    std::vector<std::vector<float>> data;
    int rows, cols;

    Matrix(int rows, int cols) : rows(rows), cols(cols){
        data.resize(rows, std::vector<float>(cols, 0.0f));    /*!!!!!!!!!!!!!!!!understand this line*/
    }
};


Matrix readPgm(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);         /*first line of the file*/
    if(line!="P2"){
        std::cerr << "Invalid PGM file " << filename << " - Magic number is not P2" << std::endl;
        exit(EXIT_FAILURE);
    }

    while(std::getline(file, line)){             /*the commented lines of the file*/
        if(line[0]!='#'){
            break;
        }
    }

    std::stringstream ss(line);            /*the line afte the commented lines of the file*/
    int cols, rows;
    if(!(ss >> cols >> rows)){
        std::cerr << "Invalid PGM file: " << filename << " - Unable to read image dimensions" << std::endl;
        exit(EXIT_FAILURE);
    }

    int max_val;
    file >> max_val;                /*next line containing the max pixel value in the file*/
    if(max_val!=255){
        std::cerr << "Invalid PGM file: " << filename << " - Max value is not 255" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix img(rows, cols);                         /*NOTE:       try recursive here */
    for (int i=0; i<rows; ++i){
        for (int j=0; j<cols; ++j){
            int pixel;
            if(!(file >> pixel)){
                std::cerr << "Invalid PGM file: " << filename << " - Unable to read pixel value" << std::endl;
                exit(EXIT_FAILURE);
            }
            img.data[i][j] = static_cast<float>(pixel) / max_val;              /*!!!!!!!!!!!!!!!!what does this line do*/
        }
    }

    file.close();
    return img;
}


float relu(float x){
    return std::max(0.0f, x);
}

std::vector<float> softmax(const std::vector<float>&  logits){
    std::vector<float> exp_vals(logits.size());
    float sum_exp = 0.0f;
    for(float val : logits){
        sum_exp += std::exp(val);
    }
    for(size_t i=0; i<logits.size(); ++i){
        exp_vals[i] = std::exp(logits[i]) / sum_exp;
    }
    return exp_vals;
}


Matrix conLayer(const Matrix& input, const Matrix& kernal){
    int outRows = input.rows - kernal.rows +1;
    int outCols = input.cols - kernal.cols +1;
    Matrix output(outRows, outCols);

    for(int i=0; i<outRows; ++i){
        for(int j=0; j<outCols; ++j){
            float sum = 0.0f;
            for(int ki=0; ki<kernal.rows; ++ki){
                for(int kj=0; kj<kernal.cols; ++kj){
                    sum += input.data[i + ki][j + kj] * kernal.data[ki][kj];
                }
            }
            output.data[i][j] = relu(sum);
        }
    }
    return output;
}


Matrix maxPooling(const Matrix& input, int poolSize){
    int outRows = input.rows / poolSize;
    int outCols = input.cols / poolSize;
    Matrix output(outRows, outCols);

    for(int i=0; i<outRows; ++i){
        for(int j=0; j<outCols; ++j){
            float maxVal = -INFINITY;
            for(int pi=0; pi<poolSize; ++pi){
                for(int pj=0; pj<poolSize; ++pj){
                    maxVal = std::max(maxVal, input.data[i * poolSize + pi][j * poolSize + pj]);
                }
            }
            output.data[i][j] = maxVal;
        }
    }
    return output;
}

float crossEtropyLoss(const std::vector<float>& predicted, int label){
    return -std::log(predicted[label]);
}


class ConLayer{
public:
    int numFilters;
    int filterSize;
    std::vector<Matrix> filters;

    ConLayer(int numFilters, int filterSize) : numFilters(numFilters), filterSize(filterSize){
        filters.resize(numFilters, Matrix(filterSize, filterSize));
        for(int filNum=0; filNum<numFilters; ++filNum){
            for(int i=0; i<filterSize; ++i){
                for(int ii=0; ii<filterSize; ++ii){
                    filters[filNum].data[i][ii] = static_cast<float>(rand() %10)/10;
                }
            }
        }
    }

    // Matrix forward(const Matrix& input){
    std::vector<Matrix> forward(const Matrix& input){
        std::vector<Matrix> finalOutput(numFilters, Matrix(input.rows, input.cols));
        for (int i=0; i<numFilters; ++i){
            finalOutput[i] = conLayer(input, filters[i]);   /*ReLU is applied inside this function*/
        }
        for (int i=0; i<numFilters; ++i){
            finalOutput[i] = maxPooling(finalOutput[i], 2);
        }
        return finalOutput;

        // Matrix out = conLayer(input, filters[0]);
        // out = maxPooling(out, 2);
        // return out;
    }
};


class FClayer{
public:
    Matrix weights;
    std::vector<float> biases;

    FClayer(int inputSize, int outSize) : weights(inputSize, outSize), biases(outSize){
        for(int i=0; i<outSize; ++i){
            biases[i] = static_cast<float>(rand() %10)/10;                     /*assigning random numbers to the weights of fullyConnected Layer*/
            for(int ii=0; ii<inputSize; ++ii){
                weights.data[ii][i] = static_cast<float>(rand() %20)/10 - 1;    /*assigning random numbers to the weights of fullyConnected Layer*/
            }
        }
    }

    std::vector<float> forward(const std::vector<float>& input){
        std::vector<float> output(biases.size());
        for(size_t i=0; i<biases.size(); ++i){
            output[i] = biases[i];
            for(size_t j=0; j<input.size(); ++j){
                output[i] += input[j] * weights.data[j][i];
            }
            output[i] = relu(output[i]);
        }
        return output;
    }
};


int main(){
    const int numFilter1=16;
    const int numFilter2=2;
    const int filterSize=5;
    srand((unsigned) time(NULL));

    std::string filename = "test.pmg";
    Matrix img = readPgm(filename);
    std::cout << "Read PGM image of size" << img.rows << "x" << img.cols << std::endl;

    ConLayer convul1 = ConLayer(numFilter1, filterSize);
    std::vector<Matrix> conImg = convul1.forward(img);

    std::vector<Matrix> conImg_final(numFilter1*numFilter2, Matrix(1, 1));
    ConLayer convul2 = ConLayer(numFilter2, filterSize);
    for(int i=0; i<numFilter1; ++i){
        std::vector<Matrix> output = convul2.forward(conImg[i]);
        conImg_final[0+ 2*i] = output[0];
        conImg_final[1+ 2*i] = output[1];
        // std::cout << 0+2*i << " " << conImg_final[0+ 2*i].data[0][0] << "\t\t" << 1+2*i << " " << conImg_final[1+ 2*i].data[0][0] << std::endl ;
    }

    // for(int i=0; i<numFilter1*numFilter2; ++i){
    //     std::cout << conImg_final[i].data[0][0] << std::endl;
    // }



    const int fcInput = conImg_final[0].cols * 32;

    FClayer fconectd = FClayer(fcInput, 10);

    std::vector<float> flattenImg(fcInput);
    // for(int fltrNum=0; fltrNum<numFilter1; ++fltrNum){
    //     for(int i=0; i<conImg[fltrNum].rows; ++i){
    //         for(int ii=0; ii<conImg[fltrNum].cols; ++ii){
    //             int index=i*6+ii;
    //             flattenImg[index] = conImg[fltrNum].data[i][ii];
    //             std::cout << std::endl << index + 36*fltrNum << " " << conImg[fltrNum].data[i][ii] << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    for(int fltrNum=0; fltrNum<numFilter2*numFilter1; ++fltrNum){
        for(int i=0; i<conImg_final[fltrNum].rows; ++i){
            for(int ii=0; ii<conImg_final[fltrNum].cols; ++ii){
                int index=i+ii;
                flattenImg[index] = conImg_final[fltrNum].data[i][ii];
                std::cout << std::endl << index + fltrNum << " " << conImg_final[fltrNum].data[i][ii] << " " << flattenImg[index] << " ";
            }
        }
    }
    std::vector<float> fc_output = fconectd.forward(flattenImg);   
    std::vector<float> fc_output_soft = softmax(fc_output);
    std::cout <<  std::endl;
    for (int i=0; i<10; ++i){
        std::cout << i << "->" << fc_output_soft[i] << "\t";
    }

    return 0;
} 