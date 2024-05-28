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
    // std::cout << rows << "x"<< cols << std::endl;

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

int final_softmax(const std::vector<float> softmax_out){
    float classifiedClass=0;
    int class_=0;
    for(int i=0; i<softmax_out.size(); ++i){
        if(std::max(softmax_out[i], classifiedClass)==softmax_out[i]){
            class_=i;
        }
        classifiedClass = std::max(softmax_out[i], classifiedClass);

        // std::cout << std::endl << std::max(softmax_out[i], classifiedClass) << std::endl; 
    }
    // std::cout << std::endl << softmax_out.size() <<std::endl;
    return class_;
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
            // std::cout << sum << relu(sum) << ":khelo " << std::endl<<std::endl;
            output.data[i][j] = relu(sum);                        /* ReLU function is applied here*/
        }
    }
    // std::cout << "khan";
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
    const int inFiltersNum;
    const int outFiltersNum;
    const int filterSize;
    const int cnnLayerNum;
    std::vector<std::vector<Matrix>> filters;

    ConLayer(const int inFiltersNum, const int outFiltersNum, int filterSize, int cnn_layerNum, std::vector<Matrix>& weights) :cnnLayerNum(cnn_layerNum), outFiltersNum(outFiltersNum), inFiltersNum(inFiltersNum), filterSize(filterSize){
        filters.resize(outFiltersNum, std::vector<Matrix>(inFiltersNum, Matrix(filterSize, filterSize)));
        // int count=0;
        for(int outFil=0; outFil<outFiltersNum; ++outFil){
            for(int inFil=0; inFil<inFiltersNum; ++inFil){

                for(int i=0; i<filterSize; ++i){
                    for(int ii=0; ii<filterSize; ++ii){

                        if(cnn_layerNum==0){
                            int index= ii + i*filterSize;
                            if(index==24){
                                // count++;
                            }
                            filters[outFil][inFil].data[i][ii] = weights[0].data[outFil][index];
                        }else if (cnn_layerNum==1){
                            int index1= ii + (i*filterSize);
                            if(index1==24){
                                // count++;
                            }
                            filters[outFil][inFil].data[i][ii] = weights[outFil].data[inFil][index1];
                            
                        }
                    }
                }

            }
        }
        // std::cout << count << std::endl;

    }

    std::vector<Matrix> forward(const std::vector<Matrix>& input){
        std::vector<Matrix> finalOutput(outFiltersNum*inFiltersNum, Matrix(input[0].rows, input[0].cols));
        for(int img=0; img<input.size(); ++img){
            for(int i=0; i<outFiltersNum; ++i){
                for(int ii=0; ii<inFiltersNum; ++ii){
                    Matrix conLayer_out = conLayer(input[img], filters[i][ii]);   /*ReLU is applied inside this function*/
                    finalOutput[ii + (i*inFiltersNum)] = maxPooling(conLayer_out, 2);
                }
            }
        }
        return finalOutput;
    }
};


class FClayer{
public:
    Matrix weights;
    std::vector<float> biases;

    FClayer(int inputSize, int outSize, const Matrix& inWeights, const Matrix& inBiases) : weights(inputSize, outSize), biases(outSize){
        for(int i=0; i<outSize; ++i){
            if(i>4){
                biases[i] = inBiases.data[1][i-5];
            }else{
                biases[i] = inBiases.data[0][i];
            }
            for(int ii=0; ii<inputSize; ++ii){

                // !!!!!Note!!!! recheck this
                weights.data[ii][i] = inWeights.data[i][ii];    /*assigning random numbers to the weights of fullyConnected Layer*/
            }
        }
    }

    std::vector<float> forward(const std::vector<float>& input, std::vector<float>& fc_output){
            std::cout<< "pass61 "<< biases.size() << std::endl;
        // std::vector<float> output;
            std::cout<< "pass62"<<std::endl;

        for(int i=0; i<biases.size(); ++i){
                std::cout<< "pass63"<<std::endl;

            fc_output[i] = biases[i];
                std::cout<< "pass63"<<std::endl;

            for(int j=0; j<input.size(); ++j){
                fc_output[i] += input[j] * weights.data[j][i];
            }
            fc_output[i] = relu(fc_output[i]);
        }
        return fc_output;
    }
};

void imgFlattener(const std::vector<Matrix>& conImg_final, std::vector<float>& flattenImg){
    for(int fltrNum=0; fltrNum<512; ++fltrNum){
        for(int i=0; i<conImg_final[fltrNum].rows; ++i){
            for(int ii=0; ii<conImg_final[fltrNum].cols; ++ii){
                
                int index = ii + i * conImg_final[fltrNum].cols + fltrNum * (conImg_final[0].rows * conImg_final[0].cols);
                flattenImg[index] = conImg_final[fltrNum].data[i][ii];
            }
        }
    }
}


int main(){
    const int inSize=1;
    const int outSize1=16;
    const int outSize2=32;
    const int filterSize=5;
    std::vector<Matrix>* cnn1Weight_pack = new std::vector<Matrix>(1, Matrix(16,25));
    std::vector<Matrix>* cnn2Weight_pack = new std::vector<Matrix>(32, Matrix(16,25));

    const Matrix cnn1_weight = readPgm("modelWeights/cnn1/cnn1_weights.pmg", 1000000);
    (*cnn1Weight_pack)[0] = cnn1_weight;

    const Matrix fc1_weight = readPgm("modelWeights/fc1/fc1_weights.pmg", 1000000);
    const Matrix fc1_bias = readPgm("modelWeights/fc1/fc1_biases.pmg", 1000000);

    for(int i=0; i<32; ++i){
        std::string filename = "modelWeights/cnn2/cnn2_weights"+ std::to_string(i) +".pmg";
        (*cnn2Weight_pack)[i] = readPgm(filename, 1000000);
    }
    std::cout<< std::endl << "pass1"<<std::endl;


    std::string filename = "2.pmg"; //"Data/finalData/trainData/13212.pmg";
    Matrix img = readPgm(filename, 255);
    std::cout << "Read PGM image of size" << img.rows << "x" << img.cols << std::endl;

    std::cout<< "pass1.5"<<std::endl;
    ConLayer convul1 = ConLayer(inSize, outSize1, filterSize, 0, (*cnn1Weight_pack));
    std::cout<< "pass2"<<std::endl;

    std::vector<Matrix>* img1 = new std::vector<Matrix>(1, Matrix(16,16));
    (*img1) = {img};
    std::vector<Matrix>* conImg = new std::vector<Matrix>(16, Matrix(8,8));
    (*conImg) = convul1.forward(*img1);

    std::vector<Matrix>* conImg_final = new std::vector<Matrix>(16*32, Matrix(4,4));
    ConLayer convul2 = ConLayer(outSize1, outSize2, filterSize, 1, (*cnn2Weight_pack));
    std::cout<< "pass3"<<std::endl;

    (*conImg_final) = convul2.forward(*conImg);

    std::cout<< "pass4"<<std::endl;

    delete cnn1Weight_pack;
    delete cnn2Weight_pack;
    delete conImg;

    const int fcInput = (*conImg_final)[0].cols * (*conImg_final)[0].rows * outSize2;
    FClayer fconectd = FClayer(fcInput, 10, fc1_weight, fc1_bias);

    std::vector<float>* flattenImg = new std::vector<float>(fcInput);
        std::cout<< "pass5"<<std::endl;
    imgFlattener(*conImg_final, *flattenImg);

    delete conImg_final;
    // for(int fltrNum=0; fltrNum<512; ++fltrNum){
    //     for(int i=0; i<conImg_final[fltrNum].rows; ++i){
    //         for(int ii=0; ii<conImg_final[fltrNum].cols; ++ii){
    //             // int index = ii + i*4 + fltrNum*16;
    //             int index = ii + i * conImg_final[fltrNum].cols + fltrNum * (conImg_final[0].rows * conImg_final[0].cols);
    //             flattenImg[index] = conImg_final[fltrNum].data[i][ii];
    //             //std::cout << index<<" "; //+ 16*fltrNum << " " << conImg_final[fltrNum].data[i][ii] << " ";
    //         }
    //     }
    //     // std::cout << std::endl<< fltrNum <<  std::endl;
    //     // break;
    // }
    std::cout<< "pass6"<<std::endl;

    // for(int i=0; i<flattenImg.size(); ++i){
    //     std::cout << flattenImg[i]<<" ";
    // }

    // std::vector<float> fc_output;
    // fconectd.forward(*flattenImg, fc_output); 

    delete flattenImg;  

        std::cout<< "pass7"<<std::endl;

    // std::vector<float> fc_output_soft = softmax(fc_output);
    // std::cout <<  std::endl;
    // for (int i=0; i<10; ++i){
    //     std::cout << i << "->" << fc_output_soft[i] << "\t";
    // }
    // std::cout<< "pass8"<<std::endl;

    // std::cout << std::endl << "the number in the img is: " << final_softmax(fc_output_soft);


    return 0;
} 