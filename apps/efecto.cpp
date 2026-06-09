#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <chrono>
#include <cstdio>

#include "EntradaEstereo.hpp"
#include "Disparidad.hpp"
#include "Profundidad.hpp"  // clases Distancia + Filtro
#include "AREffect.hpp"

using namespace cv;
using namespace std;

using Reloj = chrono::steady_clock;

static void ponerTexto(Mat& imagen, const string& texto, Point origen,
                       double escala, Scalar color, int grosor = 1) {
    int base = 0;
    Size tamanio = getTextSize(texto, FONT_HERSHEY_SIMPLEX, escala, grosor, &base);
    rectangle(imagen,
        Point(origen.x - 4, origen.y - tamanio.height - 4),
        Point(origen.x + tamanio.width + 4, origen.y + base + 4),
        Scalar(0, 0, 0), FILLED);
    putText(imagen, texto, origen, FONT_HERSHEY_SIMPLEX, escala, color, grosor);
}

int main() {
    EntradaEstereo entrada;
    if (!entrada.iniciar()) return -1;

    // Clasificador facial Haar para el efecto
    CascadeClassifier cascara;
    if (!cargarCascara(cascara, entrada.getRutaCascara())) return -1;

    Disparidad disparidad;
    Distancia  distancia(entrada.getQ());
    distancia.setDispOffset(entrada.getDispOffset());

    Filtro filtro;
    filtro.setKalmanQ(5.0);
    filtro.setKalmanR(17.0);

    namedWindow("Big Red Head",       WINDOW_NORMAL);
    namedWindow("Distancia + Filtro", WINDOW_NORMAL);
    cout << "Efecto AR en vivo. q=salir" << endl;

    double fps = 0.0;
    Reloj::time_point tPrevio = Reloj::now();

    Mat rectIzq, rectDer;
    while (true) {
        if (!entrada.leer(rectIzq, rectDer)) { waitKey(1); continue; }

        Mat    disp16    = disparidad.calcularCrudo(rectIzq, rectDer);
        double distCruda = distancia.calcular(disp16);
        Filtro::Resultado r;
        if (distCruda > 0.0) r = filtro.actualizar(distCruda);

        Mat vistaEfecto = rectIzq.clone();
        if (distCruda > 0.0)
            aplicarCabezaRoja(vistaEfecto, (float)r.distFiltrada, cascara);

        Reloj::time_point tAhora = Reloj::now();
        double dt = chrono::duration<double>(tAhora - tPrevio).count();
        tPrevio = tAhora;
        if (dt > 0.0) {
            double fpsInstantaneo = 1.0 / dt;
            fps = (fps == 0.0) ? fpsInstantaneo : (fps * 0.9 + fpsInstantaneo * 0.1);
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        ponerTexto(vistaEfecto, buf, Point(10, 25), 0.6, Scalar(0, 255, 255), 2);

        Mat vistaDist = rectIzq.clone();
        Point centro(vistaDist.cols / 2, vistaDist.rows / 2);
        Scalar cian(0, 255, 255);
        line(vistaDist, {centro.x - 20, centro.y}, {centro.x + 20, centro.y}, cian, 2);
        line(vistaDist, {centro.x, centro.y - 20}, {centro.x, centro.y + 20}, cian, 2);
        circle(vistaDist, centro, 3, cian, -1);

        if (distCruda > 0.0) {
            snprintf(buf, sizeof(buf), "raw: %.1f cm", distCruda);
            int base = 0;
            Size sz = getTextSize(buf, FONT_HERSHEY_SIMPLEX, 0.6, 1, &base);
            ponerTexto(vistaDist, buf,
                Point(vistaDist.cols - sz.width - 10, 22), 0.6, Scalar(160, 160, 160));

            snprintf(buf, sizeof(buf), "%.1f cm", r.distFiltrada);
            ponerTexto(vistaDist, buf, Point(10, 75), 1.4, Scalar(0, 255, 0), 2);
        } else {
            ponerTexto(vistaDist, "Sin datos", Point(10, 40), 0.8, Scalar(0, 0, 255));
        }

        imshow("Big Red Head",       vistaEfecto);
        imshow("Distancia + Filtro", vistaDist);

        if ((char)waitKey(1) == 'q') break;
    }

    destroyAllWindows();  
}
