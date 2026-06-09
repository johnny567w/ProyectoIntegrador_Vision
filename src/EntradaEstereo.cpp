

#include "EntradaEstereo.hpp"

#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/calib3d/calib3d.hpp>  // initUndistortRectifyMap

#include <iostream>
#include <chrono>
#include <cstdlib>  // abs

using namespace cv;
using namespace std;

using Reloj = chrono::steady_clock;
using Ms    = chrono::milliseconds;

bool EntradaEstereo::iniciar(const string& rutaConfig) {
    FileStorage configuracion(rutaConfig, FileStorage::READ);
    if (!configuracion.isOpened()) {
        cerr << "No se encontro " << rutaConfig << endl; return false;
    }
    string fuenteIzquierda, fuenteDerecha, rutaCalib;
    configuracion["cam_left"]         >> fuenteIzquierda;
    configuracion["cam_right"]        >> fuenteDerecha;
    configuracion["calib_file"]       >> rutaCalib;
    configuracion["disp_offset"]      >> dispOffset;
    configuracion["face_cascade"]     >> rutaCascara;
    configuracion.release();

    if (rutaCascara.empty())
        rutaCascara = "/home/pepe/opencv1/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";

    FileStorage archivoCalib(rutaCalib, FileStorage::READ);
    if (!archivoCalib.isOpened()) {
        cerr << "No se pudo abrir: " << rutaCalib << endl; return false;
    }
    int anchoImagen = 0, altoImagen = 0;
    Mat K1, D1, K2, D2, R1, R2, P1, P2;
    archivoCalib["image_width"]  >> anchoImagen;
    archivoCalib["image_height"] >> altoImagen;
    archivoCalib["K1"] >> K1;  archivoCalib["D1"] >> D1;
    archivoCalib["K2"] >> K2;  archivoCalib["D2"] >> D2;
    archivoCalib["R1"] >> R1;  archivoCalib["R2"] >> R2;
    archivoCalib["P1"] >> P1;  archivoCalib["P2"] >> P2;
    archivoCalib["Q"]  >> Q;  
    archivoCalib.release();

    if (anchoImagen == 0 || altoImagen == 0) {
        cerr << "Calibración inválida." << endl; return false;
    }
    if (Q.empty() || Q.rows != 4 || Q.cols != 4) {
        cerr << "La matriz Q de reproyección no está presente en " << rutaCalib << endl;
        return false;
    }
    tamanoImagen = Size(anchoImagen, altoImagen);

//mapas para corregir la camara, por si viene mal la imagen
    initUndistortRectifyMap(K1, D1, R1, P1, tamanoImagen, CV_16SC2, mapaIzqX, mapaIzqY);
    initUndistortRectifyMap(K2, D2, R2, P2, tamanoImagen, CV_16SC2, mapaDerX, mapaDerY);

    if (!camaraIzquierda.abrir(fuenteIzquierda)) return false;
    if (!camaraDerecha.abrir(fuenteDerecha))      return false;

    return true;
}

bool EntradaEstereo::leer(Mat& rectIzq, Mat& rectDer) {
    Mat fotogramaIzquierdo, fotogramaDerecho;
    Reloj::time_point marcaIzquierda, marcaDerecha;
    if (!camaraIzquierda.leer(fotogramaIzquierdo, marcaIzquierda)) return false;
    if (!camaraDerecha.leer(fotogramaDerecho, marcaDerecha))       return false;

    if (abs((int)chrono::duration_cast<Ms>(marcaIzquierda - marcaDerecha).count()) > 50)
        return false;
    if (fotogramaIzquierdo.size() != tamanoImagen)
        resize(fotogramaIzquierdo, fotogramaIzquierdo, tamanoImagen);
    if (fotogramaDerecho.size() != tamanoImagen)
        resize(fotogramaDerecho, fotogramaDerecho, tamanoImagen);

    // Rectificación
    remap(fotogramaIzquierdo, rectIzq, mapaIzqX, mapaIzqY, INTER_LINEAR);
    remap(fotogramaDerecho,   rectDer, mapaDerX, mapaDerY, INTER_LINEAR);
    return true;
}
