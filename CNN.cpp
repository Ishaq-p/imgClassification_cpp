#include "Matrix.cpp"
#include "ConLayer.cpp"
#include "FClayer.cpp"

// #include <iostream>
// #include <vector>
// #include <cmath>
#include <thread>
// #include <mutex>
#include <fstream>
#include <sstream>
// #include <atomic>
// #include <string>
// #include<cstdlib>

// std::mutex mtx; // Mutex for protecting shared data

// 219 the accuracy: 0.219
// real	0m15,456s
// user	0m15,378s
// sys	0m0,036s
const int inSize=1;
const int outSize1=16;
const int outSize2=32;
const int filterSize=5;
const int fcInput = 4 * 4 * outSize2;         // 4x4 is the frame size after the cnn2 layer
const int outClasses = 10;
const long weightsFloatingPoints = 10000000000000000L;

std::vector<Matrix> cnn1Weight_pack = std::vector<Matrix>(1, Matrix(16,25));
std::vector<Matrix> cnn2Weight_pack = std::vector<Matrix>(32, Matrix(16,25));
Matrix cnn_bias =  Matrix(2, 32);
Matrix fc1_weight =  Matrix(10, 512);
Matrix fc1_bias =  Matrix(2, 5);

ConLayer* convul1 = nullptr;
ConLayer* convul2 = nullptr;
FClayer* fconectd = nullptr;


Matrix readPgm(const std::string& filename, const long maxVal){
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

    long max_val;
    file >> max_val;                /*next line containing the max pixel value in the file*/
    if(max_val!=maxVal){
        std::cerr << "Invalid PGM file: " << filename << " - Max value is not " << maxVal << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix img(rows, cols);                         /*NOTE:       try recursive here */
    for (int i=0; i<rows; ++i){
        for (int j=0; j<cols; ++j){
            long pixel;
            if(!(file >> pixel)){
                std::cout<<pixel<<" ";
                std::cerr << "Invalid PGM file: " << filename << " - Unable to read pixel value" << std::endl;
                exit(EXIT_FAILURE);
            }
            img.data[i][j] = static_cast<double>(pixel) / static_cast<double>(max_val);              /*!!!!!!!!!!!!!!!!what does this line do*/
        }
    }

    file.close();
    return img;
}

std::vector<double> softmax(const std::vector<double>&  logits, int& class_){
    std::vector<double> exp_vals(logits.size());
    double classifiedClass=0.0f;
    class_=0;

    double sum_exp = 0.0f;
    for(double val : logits){
        sum_exp += std::exp(val);
    }
    for(size_t i=0; i<logits.size(); ++i){
        exp_vals[i] = std::exp(logits[i]) / sum_exp;
        if(std::max(logits[i], classifiedClass)==logits[i]){
            class_=i;
            classifiedClass=logits[i];
        }
        // classifiedClass = std::max(logits[i], classifiedClass);
    }
    return exp_vals;
}

double crossEtropyLoss(const std::vector<double>& predicted, int label){
    return -std::log(predicted[label]);
}

void imgFlattener(const std::vector<Matrix>& conImg_final, std::vector<double>& flattenImg){
    for(int fltrNum=0; fltrNum<32; ++fltrNum){
        for(int i=0; i<conImg_final[fltrNum].rows; ++i){
            for(int ii=0; ii<conImg_final[fltrNum].rows; ++ii){
                
                int index = ii + i * conImg_final[fltrNum].cols + fltrNum * (conImg_final[0].rows * conImg_final[0].cols);
                flattenImg[index] = conImg_final[fltrNum].data[i][ii];
            }
        }
    }
}

// void forward_processChunk(const std::vector<Matrix> img1, std::vector<Matrix>& conImg_final, const int start, const int end, const int layerNum){
//     std::vector<Matrix> output = conImg_final;
//     if(layerNum==0){
//         std::vector<Matrix> output = convul1->forward(img1, start, end);
//     }else{
//         std::vector<Matrix> output = convul2->forward(img1, start, end);
//     }
//     for(int i=start; i<end; ++i){
//         conImg_final[i] = output[i];
//     }
// }

int mini_main(const std::string& filename1){

    std::string filename = "Data/finalData/trainData/"+filename1+".pmg";
    // std::string filename = filename1+".pmg";
    std::vector<Matrix> img1 = {readPgm(filename, 10000000000000000L)};
    // std::vector<Matrix> img1 = std::vector(1, Matrix(16,16));
    // for(int i=0;i<16;++i){
    //     for(int ii=0;ii<16;++ii){
    //         img1[0].data[i][ii] = ii + (i*16);
    //     }
    // }


    std::vector<Matrix> conImg = convul1->forward(img1, 0,16);
    // for(int i=0;i<conImg[0].rows;++i){
    //         for(int ii=0;ii<conImg[0].cols;++ii){
    //             std::cout<< conImg[0].data[i][ii] << " ";
    //         }
    //         std::cout<< std::endl;
    //     }
    //                 std::cout<< std::endl;
    std::vector<Matrix> conImg_final = convul2->forward(conImg, 0, 32);
    // for(int i=0;i<conImg_final[0].rows;++i){
    //         for(int ii=0;ii<conImg_final[0].cols;++ii){
    //             std::cout<< conImg_final[0].data[i][ii] << " ";
    //         }
    //         std::cout<< std::endl;
    //     }

    // const int num_threads = 2;
    // std::vector<std::thread> threads;
    // const int chunk_size1 = 16 / num_threads;
    // for(int i=0; i<num_threads; ++i){
    //     int start = i * chunk_size1;
    //     int end = (i == num_threads - 1) ? 16 : start + chunk_size1;
    //     threads.emplace_back(forward_processChunk, std::ref(img1), std::ref(conImg), start, end, 0);  
    // }
    // for (auto &t : threads) {
    //     t.join(); // Wait for all threads to finish
    // } 

    // std::vector<std::thread> threads1;
    // const int chunk_size2 = 32 / num_threads;
    // for(int i=0; i<1; ++i){
    //     int start = i * chunk_size2;
    //     int end = (i == num_threads - 1) ? 32 : start + chunk_size2;
    //     threads1.emplace_back(forward_processChunk, std::ref(conImg), std::ref(conImg_final), start, end, 1);  
    // }
    // for (auto &t : threads1) {
    //     t.join(); // Wait for all threads to finish
    // } 

    std::vector<double> flattenImg = std::vector<double>(fcInput);
    imgFlattener(conImg_final, flattenImg);
    std::vector<double> fc_output(outClasses);
    fconectd->forward(flattenImg, fc_output); 

    int classified_class=0;
    std::vector<double> fc_output_soft = softmax(fc_output, classified_class);
    // for(int i=0; i<fc_output_soft.size(); ++i){
    //     std::cout<< i << "->" << fc_output_soft[i] << " ";
    // }
    // std::cout << std::endl << filename1 << " the number in the img is: " << classified_class;

    return classified_class;
} 



void process_chunk(const std::vector<int> &y, int start, int end, int &corrected) {
    int local_corrected = 0; // Local corrected count to avoid frequent locking
    for (int i = start; i < end; ++i) {
        std::string filename = std::to_string(i);
        int yhat = mini_main(filename);

        if (yhat == y[i]) {
            ++local_corrected;
        }
    }
    // Update the shared corrected count
    // std::lock_guard<std::mutex> lock(mtx);
    corrected += local_corrected;
}

int main(){

    cnn_bias = readPgm("modelWeights/cnn_1_2_biases.pmg", weightsFloatingPoints);
    cnn1Weight_pack[0] = readPgm("modelWeights/cnn1/cnn1_weights.pmg", weightsFloatingPoints); // readPgm("weights.pmg", 1);
    for(int i=0; i<32; ++i){
        std::string filename = "modelWeights/cnn2/cnn2_weights"+ std::to_string(i) +".pmg";
        cnn2Weight_pack[i] =  readPgm(filename, weightsFloatingPoints); // readPgm("weights.pmg", 1);
    }   
    fc1_weight = readPgm("modelWeights/fc1/fc1_weights.pmg", weightsFloatingPoints);
    fc1_bias = readPgm("modelWeights/fc1/fc1_biases.pmg", weightsFloatingPoints);

    convul1  = new ConLayer(inSize, outSize1, filterSize, 0, cnn1Weight_pack, cnn_bias);
    convul2  = new ConLayer(outSize1, outSize2, filterSize, 1, cnn2Weight_pack, cnn_bias);
    fconectd = new FClayer(fcInput, outClasses, fc1_weight, fc1_bias);
    

    // const std::vector<int> y = {};
    Matrix yValues = readPgm("yValues.pmg", weightsFloatingPoints);
    std::vector<int> y(60000);
    for(int i=0; i<yValues.rows; ++i){
        for(int ii=0; ii<yValues.cols; ++ii){
            y[ii + i*1000] = yValues.data[i][ii];
        }
    }
                        // const std::vector<int> y = {5};


    // const std::vector<int> y = {5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, 5, 9, 3, 3, 0, 7, 4, 9, 8, 0, 9, 4, 1, 4, 4, 6, 0, 4, 5, 6, 1, 0, 0, 1, 7, 1, 6, 3, 0, 2, 1, 1, 7, 9, 0, 2, 6, 7, 8, 3, 9, 0, 4, 6, 7, 4, 6, 8, 0, 7, 8, 3, 1, 5, 7, 1, 7, 1, 1, 6, 3, 0, 2, 9, 3, 1, 1, 0, 4, 9, 2, 0, 0, 2, 0, 2, 7, 1, 8, 6, 4, 1, 6, 3, 4, 5, 9, 1, 3, 3, 8, 5, 4, 7, 7, 4, 2, 8, 5, 8, 6, 7, 3, 4, 6, 1, 9, 9, 6, 0, 3, 7, 2, 8, 2, 9, 4, 4, 6, 4, 9, 7, 0, 9, 2, 9, 5, 1, 5, 9, 1, 2, 3, 2, 3, 5, 9, 1, 7, 6, 2, 8, 2, 2, 5, 0, 7, 4, 9, 7, 8, 3, 2, 1, 1, 8, 3, 6, 1, 0, 3, 1, 0, 0, 1, 7, 2, 7, 3, 0, 4, 6, 5, 2, 6, 4, 7, 1, 8, 9, 9, 3, 0, 7, 1, 0, 2, 0, 3, 5, 4, 6, 5, 8, 6, 3, 7, 5, 8, 0, 9, 1, 0, 3, 1, 2, 2, 3, 3, 6, 4, 7, 5, 0, 6, 2, 7, 9, 8, 5, 9, 2, 1, 1, 4, 4, 5, 6, 4, 1, 2, 5, 3, 9, 3, 9, 0, 5, 9, 6, 5, 7, 4, 1, 3, 4, 0, 4, 8, 0, 4, 3, 6, 8, 7, 6, 0, 9, 7, 5, 7, 2, 1, 1, 6, 8, 9, 4, 1, 5, 2, 2, 9, 0, 3, 9, 6, 7, 2, 0, 3, 5, 4, 3, 6, 5, 8, 9, 5, 4, 7, 4, 2, 7, 3, 4, 8, 9, 1, 9, 2, 8, 7, 9, 1, 8, 7, 4, 1, 3, 1, 1, 0, 2, 3, 9, 4, 9, 2, 1, 6, 8, 4, 7, 7, 4, 4, 9, 2, 5, 7, 2, 4, 4, 2, 1, 9, 7, 2, 8, 7, 6, 9, 2, 2, 3, 8, 1, 6, 5, 1, 1, 0, 2, 6, 4, 5, 8, 3, 1, 5, 1, 9, 2, 7, 4, 4, 4, 8, 1, 5, 8, 9, 5, 6, 7, 9, 9, 3, 7, 0, 9, 0, 6, 6, 2, 3, 9, 0, 7, 5, 4, 8, 0, 9, 4, 1, 2, 8, 7, 1, 2, 6, 1, 0, 3, 0, 1, 1, 8, 2, 0, 3, 9, 4, 0, 5, 0, 6, 1, 7, 7, 8, 1, 9, 2, 0, 5, 1, 2, 2, 7, 3, 5, 4, 9, 7, 1, 8, 3, 9, 6, 0, 3, 1, 1, 2, 6, 3, 5, 7, 6, 8, 3, 9, 5, 8, 5, 7, 6, 1, 1, 3, 1, 7, 5, 5, 5, 2, 5, 8, 7, 0, 9, 7, 7, 5, 0, 9, 0, 0, 8, 9, 2, 4, 8, 1, 6, 1, 6, 5, 1, 8, 3, 4, 0, 5, 5, 8, 3, 6, 2, 3, 9, 2, 1, 1, 5, 2, 1, 3, 2, 8, 7, 3, 7, 2, 4, 6, 9, 7, 2, 4, 2, 8, 1, 1, 3, 8, 4, 0, 6, 5, 9, 3, 0, 9, 2, 4, 7, 1, 2, 9, 4, 2, 6, 1, 8, 9, 0, 6, 6, 7, 9, 9, 8, 0, 1, 4, 4, 6, 7, 1, 5, 7, 0, 3, 5, 8, 4, 7, 1, 2, 5, 9, 5, 6, 7, 5, 9, 8, 8, 3, 6, 9, 7, 0, 7, 5, 7, 1, 1, 0, 7, 9, 2, 3, 7, 3, 2, 4, 1, 6, 2, 7, 5, 5, 7, 4, 0, 2, 6, 3, 6, 4, 0, 4, 2, 6, 0, 0, 0, 0, 3, 1, 6, 2, 2, 3, 1, 4, 1, 5, 4, 6, 4, 7, 2, 8, 7, 9, 2, 0, 5, 1, 4, 2, 8, 3, 2, 4, 1, 5, 4, 6, 0, 7, 9, 8, 4, 9, 8, 0, 1, 1, 0, 2, 2, 3, 2, 4, 4, 5, 8, 6, 5, 7, 7, 8, 8, 9, 7, 4, 7, 3, 2, 0, 8, 6, 8, 6, 1, 6, 8, 9, 4, 0, 9, 0, 4, 1, 5, 4, 7, 5, 3, 7, 4, 9, 8, 5, 8, 6, 3, 8, 6, 9, 9, 1, 8, 3, 5, 8, 6, 5, 9, 7, 2, 5, 0, 8, 5, 1, 1, 0, 9, 1, 8, 6, 7, 0, 9, 3, 0, 8, 8, 9, 6, 7, 8, 4, 7, 5, 9, 2, 6, 7, 4, 5, 9, 2, 3, 1, 6, 3, 9, 2, 2, 5, 6, 8, 0, 7, 7, 1, 9, 8, 7, 0, 9, 9, 4, 6, 2, 8, 5, 1, 4, 1, 5, 5, 1, 7, 3, 6, 4, 3, 2, 5, 6, 4, 4, 0, 4, 4, 6, 7, 2, 4, 3, 3, 8, 0, 0, 3, 2, 2, 9, 8, 2, 3, 7, 0, 1, 1, 0, 2, 3, 3, 8, 4, 3, 5, 7, 6, 4, 7, 7, 8, 5, 9, 7, 0, 3, 1, 6, 2, 4, 3, 4, 4, 7, 5, 9, 6, 9, 0, 7, 1, 4, 2, 7, 3, 6, 7, 5, 8, 4, 5, 5, 2, 7, 1, 1, 5, 6, 8, 5, 8, 4, 0, 7, 9, 9, 2, 9, 7, 7, 8, 7, 4, 2, 6, 9, 1, 7, 0, 6, 4, 2, 5, 7, 0, 7, 1, 0, 3, 7, 6, 5, 0, 6, 1, 5, 1, 7, 8, 5, 0, 3, 4, 7, 7, 5, 7, 8, 6, 9, 3, 8, 6, 1, 0, 9, 7, 1, 3, 0, 5, 6, 4, 4, 2, 4, 4, 3, 1, 7, 7, 6, 0, 3, 6};
    // const std::vector<int> y = {5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5};
    int corrected=0;
    double accuracy=0.0;
    const int num_threads = 8;
    const int chunk_size = y.size() / num_threads;

    std::vector<std::thread> threads;
    for(int i=0; i<num_threads; ++i){
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? y.size() : start + chunk_size;
        threads.emplace_back(process_chunk, std::ref(y), start, end, std::ref(corrected));  
    }
    for (auto &t : threads) {
        t.join(); // Wait for all threads to finish
    } 
    std::cout << corrected << " the accuracy: "<< double(corrected)/ static_cast<float> (y.size()); //float(y.size());

    delete convul1;
    delete convul2;
    delete fconectd;

}