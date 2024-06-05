// #include "Matrix.cpp"
// #include <fstream>
// #include <sstream>

class FClayer{
public:
    Matrix weights = Matrix(10,512);
    std::vector<double> biases = std::vector<double>(10);

    FClayer(int inputSize, int outSize, const Matrix& inWeights, const Matrix& inBiases) {
        for(int i=0; i<outSize; ++i){
            if(i>4){
                biases[i] = inBiases.data[1][i-5];
            }else{
                biases[i] = inBiases.data[0][i];
            }
            for(int ii=0; ii<inputSize; ++ii){
                weights.data[i][ii] = inWeights.data[i][ii];   
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input, std::vector<double>& fc_output){
        for(int i=0; i<biases.size(); ++i){
            fc_output[i] = biases[i];
            for(int j=0; j<input.size(); ++j){

                fc_output[i] += input[j] * weights.data[i][j];
            }
            fc_output[i] = relu(fc_output[i]);
        }
        return fc_output;
    }

    double relu(double x){
        return std::max(0.0, x);
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
