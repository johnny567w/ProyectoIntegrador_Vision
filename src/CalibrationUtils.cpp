#include "CalibrationUtils.hpp"

#include <filesystem>  // Gestión de directorios 
#include <algorithm>   // sort para ordenar la lista de archivos

using namespace cv;
using namespace std;
namespace fs = filesystem;

// Genera los puntos 3D del tablero: cada esquina interior en mm, Z siempre 0
vector<Point3f> crearPuntosObjeto(Size tamanoTablero, float tamCuadroMm) {
    vector<Point3f> puntos;
    puntos.reserve(tamanoTablero.area());

    for (int i = 0; i < tamanoTablero.height; i++) {
        for (int j = 0; j < tamanoTablero.width; j++) {
            puntos.emplace_back(j * tamCuadroMm, i * tamCuadroMm, 0.0f);
        }
    }
    return puntos;
}

// Lista archivos de imagen en una carpeta 
vector<string> obtenerImagenes(const string& carpeta) {
    vector<string> listado;

    for (const auto& entrada : fs::directory_iterator(carpeta)) {
        if (!entrada.is_regular_file()) continue;

        string extension = entrada.path().extension().string();
        if (extension == ".png" || extension == ".jpg" ||
            extension == ".jpeg" || extension == ".bmp") {
            listado.push_back(entrada.path().string());
        }
    }

    sort(listado.begin(), listado.end());
    return listado;
}

//de aqui spn las guias de la imagen
void dibujarGuiasHorizontales(Mat& imagen, int espaciado) {
    for (int i = espaciado; i < imagen.rows; i += espaciado) {
        line(imagen, Point(0, i), Point(imagen.cols, i), Scalar(0, 255, 0), 1);
    }
}

