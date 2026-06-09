#include <opencv2/core/core.hpp>        // Definiciones base de matrices
#include <opencv2/imgproc/imgproc.hpp>  // Procesamiento de imágenes
#include <opencv2/highgui/highgui.hpp>  // Creación de ventanas
#include <opencv2/calib3d/calib3d.hpp>  // Rectificación estéreo

#include <iostream>

#include "CameraSource.hpp"
#include "CalibrationUtils.hpp"

using namespace cv;
using namespace std;

int main(int, char**) {

    // leo config/setup para 
    FileStorage configuracion("config/setup.yml", FileStorage::READ);
    if (!configuracion.isOpened()) {
        cerr << "No se encontro config/setup.yml" << endl; return -1;
    }
    string fuenteIzquierda, fuenteDerecha, rutaCalib;
    configuracion["cam_left"]   >> fuenteIzquierda;
    configuracion["cam_right"]  >> fuenteDerecha;
    configuracion["calib_file"] >> rutaCalib;
    configuracion.release();

    // ── Leer parámetros de calibración ────────────────────────────────────────
    FileStorage archivoCalib(rutaCalib, FileStorage::READ);
    if (!archivoCalib.isOpened()) {
        cerr << "No se pudo abrir: " << rutaCalib << endl; return -1;
    }
    int anchoImagen = 0, altoImagen = 0;
    Mat K1, D1, K2, D2, R1, R2, P1, P2;
    archivoCalib["image_width"]  >> anchoImagen;
    archivoCalib["image_height"] >> altoImagen;
    archivoCalib["K1"] >> K1;  archivoCalib["D1"] >> D1;
    archivoCalib["K2"] >> K2;  archivoCalib["D2"] >> D2;
    archivoCalib["R1"] >> R1;  archivoCalib["R2"] >> R2;
    archivoCalib["P1"] >> P1;  archivoCalib["P2"] >> P2;
    archivoCalib.release();

    Size tamanoImagen(anchoImagen, altoImagen);

    // Calcular mapas de rectificación una sola vez
    Mat mapaIzqX, mapaIzqY, mapaDerX, mapaDerY;
    initUndistortRectifyMap(K1, D1, R1, P1, tamanoImagen, CV_16SC2, mapaIzqX, mapaIzqY);
    initUndistortRectifyMap(K2, D2, R2, P2, tamanoImagen, CV_16SC2, mapaDerX, mapaDerY);

    CameraSource camaraIzquierda, camaraDerecha;
    if (!camaraIzquierda.abrir(fuenteIzquierda)) return -1;
    if (!camaraDerecha.abrir(fuenteDerecha))      return -1;

    cout << "Rectificacion activa. Presiona 'q' para salir." << endl;

    while (true) {
        Mat fotogramaIzquierdo, fotogramaDerecho;

        if (!camaraIzquierda.leer(fotogramaIzquierdo) || fotogramaIzquierdo.empty()) {
            waitKey(10); continue;
        }
        if (!camaraDerecha.leer(fotogramaDerecho) || fotogramaDerecho.empty()) {
            waitKey(10); continue;
        }

        if (fotogramaIzquierdo.size() != tamanoImagen)
            resize(fotogramaIzquierdo, fotogramaIzquierdo, tamanoImagen);
        if (fotogramaDerecho.size() != tamanoImagen)
            resize(fotogramaDerecho, fotogramaDerecho, tamanoImagen);

        // Aplicar rectificación — corrige distorsión y alinea las cámaras
        Mat rectificadoIzq, rectificadoDer;
        remap(fotogramaIzquierdo, rectificadoIzq, mapaIzqX, mapaIzqY, INTER_LINEAR);
        remap(fotogramaDerecho,   rectificadoDer, mapaDerX, mapaDerY, INTER_LINEAR);

        // Guías horizontales para verificar alineación epipolar entre cámaras
        dibujarGuiasHorizontales(rectificadoIzq, 40);
        dibujarGuiasHorizontales(rectificadoDer, 40);

        // Mostrar ambas cámaras rectificadas lado a lado
        Mat vistaCompleta;
        hconcat(rectificadoIzq, rectificadoDer, vistaCompleta);
        imshow("Rectificacion estereo", vistaCompleta);

        if ((char)waitKey(1) == 'q') break;
    }

    destroyAllWindows();
    return 0;
}
