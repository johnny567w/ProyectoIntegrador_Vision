#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "EntradaEstereo.hpp"
#include "Disparidad.hpp"

using namespace cv;
using namespace std;

int main() {
    EntradaEstereo entrada;
    if (!entrada.iniciar()) return -1;

    Disparidad disparidad;

    namedWindow("Disparidad", WINDOW_NORMAL);
    cout << "Disparidad. q=salir" << endl;

    Mat rectIzq, rectDer;
    while (true) {
        if (!entrada.leer(rectIzq, rectDer)) { waitKey(1); continue; }

        Mat disp16 = disparidad.calcular(rectIzq, rectDer);
        Mat mapa   = Disparidad::colorizar(disp16);

        imshow("Disparidad", mapa);
        if ((char)waitKey(1) == 'q') break;
    }

    destroyAllWindows();  
    return 0;
}
