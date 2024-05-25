#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept> // for std::runtime_error
#include <string.h>

struct BMPFileHeader {
  unsigned short signature;
  unsigned int fileSize;
  unsigned reserved1;
  unsigned reserved2;
  unsigned int dataOffset;
};

struct BMPInfoHeader {
  unsigned int infoHeaderSize;
  int width;
  int height;
  unsigned short planes;
  unsigned short bitsPerPixel;
  unsigned int compression;
  unsigned int imageSize;
  int xPixelsPerMeter;
  int yPixelsPerMeter;
  unsigned int colorsUsed;
  unsigned int importantColors;
};

bool IsGrayscaleBMP(const BMPInfoHeader& infoHeader) {
  return infoHeader.bitsPerPixel == 8 && infoHeader.colorsUsed == 0;
}

std::vector<unsigned char> ReadBMP(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open file '" + filename + "'");
  }

  unsigned char info[54];
  file.read((char*)info, sizeof(info)); // read the 54-byte header

  if (!file.good()) {
    throw std::runtime_error("Error: Error reading file header");
  }

  BMPInfoHeader infoHeader;
  memcpy(&infoHeader, info + 14, sizeof(infoHeader)); // Extract info header

  // Verify if it's a grayscale BMP (optional)
  if (!IsGrayscaleBMP(infoHeader)) {
    std::cerr << "Warning: Image might not be grayscale format." << std::endl;
  }

  int width = infoHeader.width;
  int height = infoHeader.height;

  std::cout << std::endl;
  std::cout << "  Name: " << filename << std::endl;
  std::cout << " Width: " << width << std::endl;
  std::cout << "Height: " << height << std::endl;

  // Calculate row padding (without padding for grayscale if desired)
  int row_padded = width; // Adjust for grayscale if padding not required

  std::vector<unsigned char> data(row_padded * height);

  for (int i = 0; i < height; i++) {
    file.read((char*)data.data() + i * row_padded, row_padded);
  }

  // Optional: Channel swapping logic (comment out if not needed)
  // for (int i = 0; i < data.size(); i += 3) {
  //   unsigned char tmp = data[i];
  //   data[i] = data[i + 2];
  //   data[i + 2] = tmp;
  // }

  return data;
}

void WriteBMP(const std::string& filename, const std::vector<unsigned char>& imageData, 
             const BMPFileHeader& fileHeader, const BMPInfoHeader& infoHeader) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open file for writing: '" + filename + "'");
  }

  // Ensure header configuration for grayscale image
  BMPInfoHeader modifiedInfoHeader = infoHeader;
  modifiedInfoHeader.bitsPerPixel = 8;
  modifiedInfoHeader.compression = 0;
  modifiedInfoHeader.colorsUsed = 0;
  modifiedInfoHeader.importantColors = 0;

  // Write the modified file header with grayscale settings
  file.write((char*)&fileHeader, sizeof(fileHeader));

  // Write the modified info header
  file.write((char*)&modifiedInfoHeader, sizeof(modifiedInfoHeader));

  // Write the image data
  file.write((char*)imageData.data(), imageData.size());

  file.close();
  std::cout << "Image data written to: " << filename << std::endl;
}


int main() {
  std::string filename = "sample_image.bmp"; // Replace with your image filename
  std::vector<unsigned char> imageData;

  try {
    imageData = ReadBMP(filename);
    std::cout << "Image data read successfully!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Extract header information from the already read data (assuming ReadBMP populates these structs)
  BMPFileHeader fileHeader = *(BMPFileHeader*)imageData.data();
  BMPInfoHeader infoHeader = *(BMPInfoHeader*)(imageData.data() + sizeof(BMPFileHeader));

  // You can now process the image data (optional)

  // Save the image with a new filename (optional)
  std::string outputFilename = "saved_image.bmp";
  try {
    WriteBMP(outputFilename, imageData, fileHeader, infoHeader);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

// int main() {
//   std::string filename = "./Data/finalData/trainData/1.bmp"; // Replace with your image filename
//   std::vector<unsigned char> imageData;

//   try {
//     imageData = ReadBMP(filename);
//     std::cout << "Image data read successfully!" << std::endl;
//   } catch (const std::exception& e) {
//     std::cerr << "Error: " << e.what() << std::endl;
//   }

//   for (int i=0; i<imageData.size(); i++){
//     std::cout << imageData[i] <<"\t";
//   }
//   // You can now process the image data stored in the vector 'imageData'

//   return 0;
// }
