#include <opencv2/core/core.hpp>        // Definiciones base de matrices
#include <opencv2/imgproc/imgproc.hpp>  // Procesamiento de imágenes
#include <opencv2/highgui/highgui.hpp>  // Creación de ventanas
#include <opencv2/imgcodecs/imgcodecs.hpp> // Guardar imágenes
#include <opencv2/calib3d/calib3d.hpp>  // Detección de tablero de calibración

#include <iostream>
#include <filesystem>  // Gestión de directorios 
#include <iomanip>
#include <sstream>

#include "CameraSource.hpp"

using namespace cv;
using namespace std;
namespace fs = filesystem;

int main(int, char**) {
    //  Leer configuración desde setup.yml 
    FileStorage configuracion("config/setup.yml", FileStorage::READ);
    if (!configuracion.isOpened()) {
        cerr << "No se encontro config/setup.yml" << endl; return -1;
    }
    string fuenteIzquierda, fuenteDerecha, carpetaImagenes;
    int anchoTablero = 0, altoTablero = 0;
    configuracion["cam_left"]     >> fuenteIzquierda;
    configuracion["cam_right"]    >> fuenteDerecha;
    configuracion["board_width"]  >> anchoTablero;
    configuracion["board_height"] >> altoTablero;
    configuracion["calib_images"] >> carpetaImagenes;
    configuracion.release();

    if (fuenteIzquierda.empty() || fuenteDerecha.empty()) {
        cerr << "cam_left / cam_right no definidas en config/setup.yml" << endl; return -1;
    }
    if (anchoTablero < 2 || altoTablero < 2) {
        cerr << "board_width / board_height invalidos en config/setup.yml" << endl; return -1;
    }

    cout << "cam_left:  " << fuenteIzquierda << endl;
    cout << "cam_right: " << fuenteDerecha   << endl;
    cout << "Tablero:   " << anchoTablero << " x " << altoTablero << " esquinas internas" << endl;

    Size tamanoTablero(anchoTablero, altoTablero);

    // apunta a donde iran las imagenes
    fs::create_directories(carpetaImagenes + "/left");
    fs::create_directories(carpetaImagenes + "/right");

    CameraSource camaraIzquierda, camaraDerecha;
    if (!camaraIzquierda.abrir(fuenteIzquierda)) return -1;
    if (!camaraDerecha.abrir(fuenteDerecha))      return -1;

    int contador = 0;

    cout << "Captura imagenes"      << endl;
    cout << "Presiona 'c' para guardar un par de imagenes." << endl;
    cout << "Presiona 'q' para salir."                      << endl;

    while (true) {
        Mat fotogramaIzquierdo, fotogramaDerecho;

        if (!camaraIzquierda.leer(fotogramaIzquierdo) || fotogramaIzquierdo.empty()) {
            waitKey(10); continue;
        }
        if (!camaraDerecha.leer(fotogramaDerecho) || fotogramaDerecho.empty()) {
            waitKey(10); continue;
        }

        // Convertir a escala de grises 
        Mat grisIzquierdo, grisDerecho;
        cvtColor(fotogramaIzquierdo, grisIzquierdo, COLOR_BGR2GRAY);
        cvtColor(fotogramaDerecho,   grisDerecho,   COLOR_BGR2GRAY);

        // Buscar esquinas del tablero en ambas cámaras
        vector<Point2f> esquinasIzquierda, esquinasDerecha;

        bool encontradoIzq = findChessboardCorners(grisIzquierdo, tamanoTablero, esquinasIzquierda,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        bool encontradoDer = findChessboardCorners(grisDerecho, tamanoTablero, esquinasDerecha,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        // Dibujar esquinas detectadas sobre copias de los fotogramas
        Mat vistaIzquierda = fotogramaIzquierdo.clone();
        Mat vistaDerecha   = fotogramaDerecho.clone();

        if (encontradoIzq) drawChessboardCorners(vistaIzquierda, tamanoTablero, esquinasIzquierda, encontradoIzq);
        if (encontradoDer) drawChessboardCorners(vistaDerecha,   tamanoTablero, esquinasDerecha,   encontradoDer);

        putText(vistaIzquierda,
            encontradoIzq ? "IZQ OK" : "IZQ NO",
            Point(20, 35), FONT_HERSHEY_SIMPLEX, 1.0,
            encontradoIzq ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);

        putText(vistaDerecha,
            encontradoDer ? "DER OK" : "DER NO",
            Point(20, 35), FONT_HERSHEY_SIMPLEX, 1.0,
            encontradoDer ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);

        // Mostrar ambas cámaras 
        Mat vistaCompleta;
        hconcat(vistaIzquierda, vistaDerecha, vistaCompleta);
        imshow("Captura imagenes", vistaCompleta);

        char tecla = (char)waitKey(1);
        if (tecla == 'q') break;

        if (tecla == 'c') {
            if (!encontradoIzq || !encontradoDer) {
                cout << "No se guardo. El tablero debe detectarse en ambas camaras." << endl;
                continue;
            }

            // Construir nombres de archivo con número de secuencia 
            ostringstream nombreIzq, nombreDer;
            nombreIzq << carpetaImagenes << "/left/left_"
                      << setw(3) << setfill('0') << contador << ".png";
            nombreDer << carpetaImagenes << "/right/right_"
                      << setw(3) << setfill('0') << contador << ".png";

            imwrite(nombreIzq.str(), fotogramaIzquierdo);
            imwrite(nombreDer.str(), fotogramaDerecho);

            cout << "Par guardado: " << contador << endl;
            contador++;
        }
    }

    destroyAllWindows();
    return 0;
}
