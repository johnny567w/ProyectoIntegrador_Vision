#include <opencv2/core/core.hpp>        // Definiciones base de matrices
#include <opencv2/imgproc/imgproc.hpp>  // Procesamiento de imágenes
#include <opencv2/imgcodecs/imgcodecs.hpp> // Leer imágenes
#include <opencv2/calib3d/calib3d.hpp>  // Calibración estéreo

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "CalibrationUtils.hpp"

using namespace cv;
using namespace std;

// Calcula el error de reproyección por imagen individual
static double calcularErrorImagen(
    const Mat& K, const Mat& D,
    const vector<Point3f>& puntosObjeto,
    const vector<Point2f>& puntosImagen,
    const Mat& rvec, const Mat& tvec)
{
    vector<Point2f> puntosProyectados;
    projectPoints(puntosObjeto, rvec, tvec, K, D, puntosProyectados);
    double suma = 0;
    for (size_t i = 0; i < puntosImagen.size(); i++) {
        double dx = puntosImagen[i].x - puntosProyectados[i].x;
        double dy = puntosImagen[i].y - puntosProyectados[i].y;
        suma += dx*dx + dy*dy;
    }
    return sqrt(suma / puntosImagen.size());
}

int main(int, char**) {
//configuracion de setupyml, solo lee
    FileStorage configuracion("config/setup.yml", FileStorage::READ);
    if (!configuracion.isOpened()) {
        cerr << "No se encontro config/setup.yml" << endl; return -1;
    }
    string carpetaImagenes, archivoSalida;
    int anchoTablero = 0, altoTablero = 0;
    float tamCuadroMm = 0.0f;
    configuracion["calib_images"]   >> carpetaImagenes;
    configuracion["calib_file"]     >> archivoSalida;
    configuracion["board_width"]    >> anchoTablero;
    configuracion["board_height"]   >> altoTablero;
    configuracion["square_size_mm"] >> tamCuadroMm;
    configuracion.release();

    if (carpetaImagenes.empty() || archivoSalida.empty()) {
        cerr << "calib_images / calib_file no definidos en config/setup.yml" << endl; return -1;
    }
    if (anchoTablero < 2 || altoTablero < 2 || tamCuadroMm <= 0) {
        cerr << "board_width, board_height o square_size_mm invalidos" << endl; return -1;
    }

    cout << "Carpeta imagenes: " << carpetaImagenes << endl;
    cout << "Tablero: " << anchoTablero << " x " << altoTablero
         << "  cuadro: " << tamCuadroMm << " mm" << endl;
    cout << "Salida:  " << archivoSalida << endl << endl;

    Size tamanoTablero(anchoTablero, altoTablero);

    // Obtener listas de imágenes 
    vector<string> imagenesIzquierda = obtenerImagenes(carpetaImagenes + "/left");
    vector<string> imagenesDerecha   = obtenerImagenes(carpetaImagenes + "/right");

    if (imagenesIzquierda.empty() || imagenesDerecha.empty()) {
        cerr << "No hay imagenes suficientes para calibrar." << endl; return -1;
    }
    if (imagenesIzquierda.size() != imagenesDerecha.size()) {
        cerr << "La cantidad de imagenes izquierda y derecha no coincide." << endl; return -1;
    }

    vector<vector<Point2f>> puntosImagenIzq, puntosImagenDer;
    vector<vector<Point3f>> puntosObjeto;
    vector<Point3f> puntosBase = crearPuntosObjeto(tamanoTablero, tamCuadroMm);

    Size tamanoImagen;

    const Size  ventanaSubPix(5, 5);
    const TermCriteria criterioSubPix(TermCriteria::EPS + TermCriteria::MAX_ITER, 50, 1e-5);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));

    int aceptados = 0, omitidos = 0;

    for (size_t i = 0; i < imagenesIzquierda.size(); i++) {
        Mat izquierda = imread(imagenesIzquierda[i]);
        Mat derecha   = imread(imagenesDerecha[i]);

        if (izquierda.empty() || derecha.empty()) { omitidos++; continue; }

        tamanoImagen = izquierda.size();

        // Convertir a grises 
        Mat grisIzq, grisDer;
        cvtColor(izquierda, grisIzq, COLOR_BGR2GRAY);
        cvtColor(derecha,   grisDer, COLOR_BGR2GRAY);

        // CLAHE mejora el contraste para detectar esquinas en iluminación desigual
        clahe->apply(grisIzq, grisIzq);
        clahe->apply(grisDer, grisDer);

        vector<Point2f> esquinasIzq, esquinasDer;

        bool encontradoIzq = findChessboardCorners(grisIzq, tamanoTablero, esquinasIzq,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        bool encontradoDer = findChessboardCorners(grisDer, tamanoTablero, esquinasDer,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        if (encontradoIzq && encontradoDer) {
            cornerSubPix(grisIzq, esquinasIzq, ventanaSubPix, Size(-1,-1), criterioSubPix);
            cornerSubPix(grisDer, esquinasDer, ventanaSubPix, Size(-1,-1), criterioSubPix);

            puntosImagenIzq.push_back(esquinasIzq);
            puntosImagenDer.push_back(esquinasDer);
            puntosObjeto.push_back(puntosBase);
            cout << "Par valido: " << i << endl;
            aceptados++;
        } else {
            cout << "Par ignorado (no detectado): " << i << endl;
            omitidos++;
        }
    }

    cout << endl << "Pares aceptados: " << aceptados << "  omitidos: " << omitidos << endl;

    if ((int)puntosImagenIzq.size() < 10) {
        cerr << "Pares validos insuficientes (" << puntosImagenIzq.size()
             << "). Se necesitan al menos 10." << endl;
        return -1;
    }

    const int flagsCalib = CALIB_FIX_K3;

    Mat K1, D1, K2, D2;
    vector<Mat> rvecs1, tvecs1, rvecs2, tvecs2;

    double errorIzquierda = calibrateCamera(puntosObjeto, puntosImagenIzq,
        tamanoImagen, K1, D1, rvecs1, tvecs1, flagsCalib);
    double errorDerecha   = calibrateCamera(puntosObjeto, puntosImagenDer,
        tamanoImagen, K2, D2, rvecs2, tvecs2, flagsCalib);

    // descarga imagenes mala que tengan medida > 1.6
    vector<double> erroresPorImagen(puntosImagenIzq.size());
    for (size_t i = 0; i < puntosImagenIzq.size(); i++)
        erroresPorImagen[i] = calcularErrorImagen(K1, D1, puntosObjeto[i],
                                                  puntosImagenIzq[i], rvecs1[i], tvecs1[i]);

    vector<double> ordenados = erroresPorImagen;
    sort(ordenados.begin(), ordenados.end());
    double mediana   = ordenados[ordenados.size() / 2];
    double umbral    = max(1.5 * mediana, 1.0);

    vector<vector<Point2f>> filtradoIzq, filtradoDer;
    vector<vector<Point3f>> filtradoObj;
    int descartados = 0;

    for (size_t i = 0; i < puntosImagenIzq.size(); i++) {
        if (erroresPorImagen[i] <= umbral) {
            filtradoIzq.push_back(puntosImagenIzq[i]);
            filtradoDer.push_back(puntosImagenDer[i]);
            filtradoObj.push_back(puntosObjeto[i]);
        } else {
            cout << "Par descartado (outlier): " << i
                 << "  error=" << erroresPorImagen[i] << endl;
            descartados++;
        }
    }

    if (descartados > 0) {
        cout << "Outliers descartados: " << descartados << ". Recalibrando..." << endl;
        errorIzquierda = calibrateCamera(filtradoObj, filtradoIzq, tamanoImagen,
                                         K1, D1, rvecs1, tvecs1, flagsCalib);
        errorDerecha   = calibrateCamera(filtradoObj, filtradoDer, tamanoImagen,
                                         K2, D2, rvecs2, tvecs2, flagsCalib);
    }

    Mat R, T, E, F;
    double errorEstereo = stereoCalibrate(
        filtradoObj, filtradoIzq, filtradoDer,
        K1, D1, K2, D2, tamanoImagen, R, T, E, F,
        CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_K3,
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 1e-6));

    Mat R1, R2, P1, P2, Q;
    stereoRectify(K1, D1, K2, D2, tamanoImagen,
                  R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 0, tamanoImagen);

    // Guardar todos los parámetros en el archivo de calibración
    FileStorage fs(archivoSalida, FileStorage::WRITE);
    fs << "image_width"    << tamanoImagen.width;
    fs << "image_height"   << tamanoImagen.height;
    fs << "board_width"    << anchoTablero;
    fs << "board_height"   << altoTablero;
    fs << "square_size_mm" << tamCuadroMm;
    fs << "K1" << K1 << "D1" << D1;
    fs << "K2" << K2 << "D2" << D2;
    fs << "R"  << R  << "T"  << T  << "E" << E << "F" << F;
    fs << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
    fs << "error_left"   << errorIzquierda;
    fs << "error_right"  << errorDerecha;
    fs << "error_stereo" << errorEstereo;
    fs.release();

    cout << endl << "Calibracion terminada." << endl;
    cout << "Error camara izquierda: " << errorIzquierda << endl;
    cout << "Error camara derecha:   " << errorDerecha   << endl;
    cout << "Error estereo:          " << errorEstereo   << endl;
    cout << "Archivo guardado en: "    << archivoSalida  << endl;

    return 0;
}
