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


Matrix readPgm(const std::string& filename, const int maxVal){
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
    if(!(ss >> rows >> cols)){
        std::cerr << "Invalid PGM file: " << filename << " - Unable to read image dimensions" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << rows << cols << std::endl;

    int max_val;
    file >> max_val;                /*next line containing the max pixel value in the file*/
    if(max_val!=maxVal){
        std::cerr << "Invalid PGM file: " << filename << " - Max value is not" << maxVal << std::endl;
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

Matrix addPadding(const Matrix& input, const int paddingSize){
    Matrix paddedInput = Matrix(input.rows+paddingSize*2, input.cols+paddingSize*2);
    for(int i=0; i<input.rows+paddingSize*2; ++i){                   /*since we have to apply two extra lines to each side, hence we have to multiply to 2*/
        for(int ii=0; ii<input.cols+paddingSize*2; ++ii){
            if((i>1 && ii>1) && (i<input.rows && ii<input.cols)){
                paddedInput.data[i][ii] = input.data[i][ii];
            }else{
                paddedInput.data[i][ii] = 0;
            }
        }
    }
    return paddedInput;
}


Matrix conLayer(const Matrix& input, const Matrix& kernal){     /* KernalSize(5,5),  stride=1, padding=(2,2)*/
    Matrix paddedInput = addPadding(input, 2);
    int outRows = paddedInput.rows - kernal.rows +1;
    int outCols = paddedInput.cols - kernal.cols +1;
    Matrix output(outRows, outCols);

    for(int i=0; i<outRows; ++i){
        for(int j=0; j<outCols; ++j){
            float sum = 0.0f;
            for(int ki=0; ki<kernal.rows; ++ki){
                for(int kj=0; kj<kernal.cols; ++kj){
                    sum += paddedInput.data[i + ki][j + kj] * kernal.data[ki][kj];
                }
            }
            output.data[i][j] = relu(sum);                        /* ReLU function is applied here*/
        }
    }
    return output;
}


Matrix maxPooling(const Matrix& input, int poolSize){    /* KernalSize(2,2),  stride=2, padding=(0,0)*/
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

    ConLayer(int numFilters, int filterSize, int cnn_layerNum, std::vector<Matrix>& weights) : numFilters(numFilters), filterSize(filterSize){
        filters.resize(numFilters, Matrix(filterSize, filterSize));
        for(int filNum=0; filNum<numFilters; ++filNum){
            for(int i=0; i<filterSize; ++i){
                for(int ii=0; ii<filterSize; ++ii){

                    if(cnn_layerNum==0){
                        filters[filNum].data[i][ii] = weights[0].data[filNum][ii + 5*i];
                    }else{
                    filters[filNum].data[i][ii] = static_cast<float>(rand() %10)/10;
                    }
                }
            }
        }
    }

    std::vector<Matrix> forward(const Matrix& input){
        std::vector<Matrix> finalOutput(numFilters, Matrix(input.rows, input.cols));
        for (int i=0; i<numFilters; ++i){
            Matrix conLayer_out = conLayer(input, filters[i]);   /*ReLU is applied inside this function*/
            finalOutput[i] = maxPooling(conLayer_out, 2);
        }
        return finalOutput;
    }

    // void storeWeights(std::vector<Matrix>& weights){
    //     for(size_t i; i<filters.size(); ++i){

    //     }

    // }
};


class FClayer{
public:
    Matrix weights;
    std::vector<float> biases;

    FClayer(int inputSize, int outSize, const Matrix& inWeights) : weights(inputSize, outSize), biases(outSize){
        for(int i=0; i<outSize; ++i){
            biases[i] = static_cast<float>(rand() %10)/10;                     /*assigning random numbers to the weights of fullyConnected Layer*/
            for(int ii=0; ii<inputSize; ++ii){

                // !!!!!Note!!!! recheck this
                weights.data[ii][i] = inWeights.data[i][ii];    /*assigning random numbers to the weights of fullyConnected Layer*/
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
    const int numFilter2=32;
    const int filterSize=5;
    srand((unsigned) time(NULL));
    std::vector<Matrix> cnn1Weight_pack(1, Matrix(16,25));
    std::vector<Matrix> cnn2Weight_pack(32, Matrix(16,25));

    const Matrix cnn1_weight = readPgm("modelWeights/cnn1/cnn1_weights.pmg", 1000000);
    cnn1Weight_pack[0] = cnn1_weight;

    const Matrix fc1_weight = readPgm("modelWeights/fc1/fc1_weights.pmg", 1000000);

    for(int i=0; i<32; ++i){
        std::string filename = "modelWeights/cnn2/cnn2_weights"+ std::to_string(i) +".pmg";
        cnn2Weight_pack[i] = readPgm(filename, 1000000);
    }


    std::string filename = "test.pmg";
    Matrix img = readPgm(filename, 255);
    std::cout << "Read PGM image of size" << img.rows << "x" << img.cols << std::endl;

    ConLayer convul1 = ConLayer(numFilter1, filterSize, 0, cnn1Weight_pack);
    std::vector<Matrix> conImg = convul1.forward(img);

    std::vector<Matrix> conImg_final(numFilter1*numFilter2, Matrix(1, 1));
    ConLayer convul2 = ConLayer(numFilter2, filterSize, 1, cnn2Weight_pack);
    for(int i=0; i<numFilter1; ++i){
        std::vector<Matrix> output = convul2.forward(conImg[i]);
        conImg_final[0+ 2*i] = output[0];
        conImg_final[1+ 2*i] = output[1];
        // std::cout << 0+2*i << " " << conImg_final[0+ 2*i].rows << "\t\t" << 1+2*i << " " << conImg_final[1+ 2*i].cols << std::endl ;
    }


    const int fcInput = conImg_final[0].cols * conImg_final[0].rows * numFilter2;

    FClayer fconectd = FClayer(fcInput, 10, fc1_weight);

    std::vector<float> flattenImg(fcInput);
    for(int fltrNum=0; fltrNum<numFilter2; ++fltrNum){
        for(int i=0; i<conImg_final[fltrNum].rows; ++i){
            for(int ii=0; ii<conImg_final[fltrNum].cols; ++ii){
                int index=i*conImg_final[fltrNum].rows+ii;
                flattenImg[index] = conImg_final[fltrNum].data[i][ii];
                // std::cout << std::endl << index + 16*fltrNum << " " << conImg_final[fltrNum].data[i][ii] << " ";
            }
        }
        // std::cout << std::endl;
        // break;
    }

    std::vector<float> fc_output = fconectd.forward(flattenImg);   
    std::vector<float> fc_output_soft = softmax(fc_output);
    std::cout <<  std::endl;
    for (int i=0; i<10; ++i){
        std::cout << i << "->" << fc_output_soft[i] << "\t";
    }


    return 0;
} 