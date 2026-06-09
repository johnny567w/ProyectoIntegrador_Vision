#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <cstdio>

#include "EntradaEstereo.hpp"
#include "Disparidad.hpp"
#include "Profundidad.hpp"  // clases Distancia + Filtro

using namespace cv;
using namespace std;

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

    Disparidad disparidad;
    Distancia  distancia(entrada.getQ());
    distancia.setDispOffset(entrada.getDispOffset());

    Filtro filtro;
    filtro.setKalmanQ(5.0);
    filtro.setKalmanR(17.0);

    printf("=== Parámetros Estéreo ===\n");
    printf("  focal=%.1fpx  base=%.3fcm  disp_offset=%.1fpx\n",
        distancia.getFocalPx(), distancia.getBaselineCm(), entrada.getDispOffset());

    namedWindow("Distancia + Filtro", WINDOW_NORMAL);
    cout << "Midiendo" << endl;

    Mat rectIzq, rectDer;

    while (true) {
        if (!entrada.leer(rectIzq, rectDer)) { waitKey(1); continue; }

        Mat    disp16    = disparidad.calcularCrudo(rectIzq, rectDer);
        double distCruda = distancia.calcular(disp16);

        Mat vista = rectIzq.clone();
        Point centro(vista.cols / 2, vista.rows / 2);

        Scalar cian(0, 255, 255);
        line(vista, {centro.x - 20, centro.y}, {centro.x + 20, centro.y}, cian, 2);
        line(vista, {centro.x, centro.y - 20}, {centro.x, centro.y + 20}, cian, 2);
        circle(vista, centro, 3, cian, -1);

        if (distCruda > 0.0) {
            Filtro::Resultado r = filtro.actualizar(distCruda);
            char buf[64];

            snprintf(buf, sizeof(buf), "raw: %.1f cm", distCruda);
            int base = 0;
            Size sz = getTextSize(buf, FONT_HERSHEY_SIMPLEX, 0.6, 1, &base);
            ponerTexto(vista, buf,
                Point(vista.cols - sz.width - 10, 22), 0.6, Scalar(160, 160, 160));

//vere el filtro kalman
                snprintf(buf, sizeof(buf), "%.1f cm", r.distFiltrada);
            ponerTexto(vista, buf, Point(10, 75), 1.4, Scalar(0, 255, 0), 2);

        } else {
            ponerTexto(vista, "Sin datos", Point(10, 40), 0.8, Scalar(0, 0, 255));
        }

        imshow("Distancia + Filtro", vista);

        if ((char)waitKey(1) == 'q') break;
    }

    destroyAllWindows();  
    return 0;
}
