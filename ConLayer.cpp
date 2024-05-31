// #include "Matrix.cpp"
#include <cmath>
// #include <fstream>
// #include <sstream>


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


Matrix convolve2d(const Matrix& X, const Matrix& kernel, int stride = 1, int padding = 2) {
    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;
    int output_height = (X.rows - kernel_height + 2 * padding) / stride + 1;
    int output_width = (X.cols - kernel_width + 2 * padding) / stride + 1;

    Matrix output(output_height, output_width);

    Matrix X_padded = Matrix(X.rows + 2 * padding, X.cols + 2 * padding);
    for (int i = 0; i < X.rows; ++i) {
        for (int j = 0; j < X.cols; ++j) {
            X_padded.data[i + padding][j + padding] = X.data[i][j];
        }
    }

    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            double sum = 0.0f;
            for (int i = 0; i < kernel_height; ++i) {
                for (int j = 0; j < kernel_width; ++j) {
                    sum += X_padded.data[y * stride + i][x * stride + j] * kernel.data[i][j];
                }
            }
            output.data[y][x] = sum;
        }
    }
    return output;
}


double relu(double x){
    return std::max(0.0, x);
}

Matrix addPadding(const Matrix& input, const int paddingSize){
    Matrix paddedInput = Matrix(input.rows+paddingSize*2, input.cols+paddingSize*2);
    for(int i=0; i<input.rows+paddingSize*2; ++i){                   /*since we have to apply two extra lines to each side, hence we have to multiply to 2*/
        for(int ii=0; ii<input.cols+paddingSize*2; ++ii){
            if((i>=paddingSize && i<input.cols+2) & (ii>=paddingSize && ii<input.cols+2)){
                paddedInput.data[i][ii] = input.data[i-2][ii-2];
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
            double sum = 0.0f;
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
            double maxVal = -INFINITY;
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

class ConLayer {
public:
    const int inFiltersNum;
    const int outFiltersNum;
    const int filterSize;
    const int cnnLayerNum;
    std::vector<std::vector<Matrix>> filters;
    Matrix cnn_bias;

    ConLayer(const int inFiltersNum, const int outFiltersNum, int filterSize, int cnnLayerNum, std::vector<Matrix>& weights, const Matrix& cnn_bias) 
        : inFiltersNum(inFiltersNum), outFiltersNum(outFiltersNum), filterSize(filterSize), cnnLayerNum(cnnLayerNum), cnn_bias(cnn_bias) {
       
        filters.resize(outFiltersNum, std::vector<Matrix>(inFiltersNum, Matrix(filterSize, filterSize)));
        
        for(int outFil = 0; outFil < outFiltersNum; ++outFil) {
            
            for(int inFil = 0; inFil < inFiltersNum; ++inFil) {

                for(int i = 0; i < filterSize; ++i) {
                    for(int ii = 0; ii < filterSize; ++ii) {
                        int index=0;
                        if (cnnLayerNum == 0) {
                            index = ii + i * filterSize;
                            filters[outFil][inFil].data[i][ii] = weights[0].data[outFil][index];
                        } else if (cnnLayerNum == 1) {
                            index = ii + i * filterSize;
                            filters[outFil][inFil].data[i][ii] = weights[outFil].data[inFil][index];
                        }
                    
                    }
                }
            }
        }
    }


    std::vector<Matrix> forward(const std::vector<Matrix>& input) {
        std::vector<Matrix> finalOutput(outFiltersNum, Matrix(  (input[0].rows - filterSize +1 +4)/2, // +4 is for padding, since we add padding to both start and end hence it will be *2
                                                                (input[0].cols - filterSize +1 +4)/2));
        
        Matrix temp((input[0].rows - filterSize +1 +4),
                    (input[0].cols - filterSize +1 +4)); // Temp matrix to accumulate results      

        for(int img = 0; img < input.size(); ++img) {

            for(int i = 0; i < outFiltersNum; ++i) {
                temp.set2zero();
                for(int ii = 0; ii < inFiltersNum; ++ii) {
                    Matrix conLayer_out = conLayer(input[img], filters[i][ii]); // ReLU applied inside conLayer

                    for(int k=0; k<temp.cols; ++k){                          // same as assigning temp += conLayer_out
                        for(int kk=0; kk<temp.cols; ++kk){
                            temp.data[k][kk] += conLayer_out.data[k][kk]; // Accumulate the results
                        }
                    }

                }
                // if(cnnLayerNum==1){
                // for(int k=0; k<temp.cols; ++k){
                //     for(int kk=0; kk<temp.cols; ++kk){
                //         temp.data[k][kk] += cnn_bias.data[cnnLayerNum][i]; // Accumulate the results
                //     }
                // }
                // }
                finalOutput[i] = maxPooling(temp, 2); // Apply max pooling on the accumulated results
            }
        }
        return finalOutput;
    }

    void printWeights(){
        for(int k=0; k<filters.size(); ++k){
            for(int kk=0; kk<filters[k].size(); ++kk){
                for(int i=0;i<filters[k][kk].rows;++i){
                    for(int ii=0;ii<filters[k][kk].cols;++ii){
                        std::cout<< filters[k][kk].data[i][ii] <<" ";//* 1000000 << " ";
                        std::cout.precision(16);
                    }
                    std::cout<< std::endl;
                }
                std::cout<< std::endl;
            }
            std::cout<< std::endl;
        }
    }
};

// int main(){
//     std::vector<Matrix> cnn1Weight_pack = std::vector<Matrix>(3, Matrix(16,25));
//     // const Matrix cnn_bias = readPgm("modelWeights/cnn_1_2_biases.pmg", 1000000);
//     const Matrix cnn_bias = readPgm("biases.pmg", 1);
//     // const Matrix cnn1_weight = readPgm("modelWeights/cnn1/cnn1_weights.pmg", 1000000);
//     const Matrix cnn1_weight = readPgm("weights.pmg", 1);
//     cnn1Weight_pack[0] = cnn1_weight;


//     std::vector<Matrix> img = std::vector(1, Matrix(16,16));
//     for(int i=0;i<16;++i){
//         for(int ii=0;ii<16;++ii){
//             img[0].data[i][ii] = 1;
//         }
//     }

//     ConLayer con = ConLayer(1, 16, 5, 0, cnn1Weight_pack, cnn_bias);
//     std::vector<Matrix> out = con.forward(img);
//     // con.printWeights();

//     // for(int k=0; k<out.size(); ++k){
//         for(int i=0;i<out[0].rows;++i){
//             for(int ii=0;ii<out[0].cols;++ii){
//                 std::cout<< out[0].data[i][ii] << " ";
//             }
//             std::cout<< std::endl;
//         }
//     //     std::cout<< std::endl;
//     // }
//     std::cout << "hello";
// }



