#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// using std::string;

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

int main(){
    std::string filename = "test.pmg";
    Matrix img = readPgm(filename);

    std::cout << "Read PGM image of size" << img.rows << "x" << img.cols << std::endl;

    for(int i=0; i<img.rows; ++i){
        for (int j=0; j<img.cols; ++j){
            std::cout << img.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}