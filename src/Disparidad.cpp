#include "Disparidad.hpp"

#include <opencv2/imgproc/imgproc.hpp>  // cvtColor, GaussianBlur, CLAHE

using namespace cv;
    // SGBM: calcula el mapa de disparidad
    // WLS:  suaviza el mapa conservando bordes
Disparidad::Disparidad() {

    clahe = createCLAHE(2.0, Size(8, 8));

    matcherIzquierdo = StereoSGBM::create(
        0, 128, 9, 648,  2592, 2, 63,8,200,32,   
        StereoSGBM::MODE_SGBM_3WAY);

    matcherDerecho = ximgproc::createRightMatcher(matcherIzquierdo);
    filtroWLS      = ximgproc::createDisparityWLSFilter(matcherIzquierdo);
    filtroWLS->setLambda(6000.0);
    filtroWLS->setSigmaColor(1.5);
}

void Disparidad::preprocesar(const Mat& rectIzq, const Mat& rectDer,
                             Mat& outIzq, Mat& outDer) {
    // clahe con escalado en gris, primero se le hace gris
    Mat grisIzq, grisDer;
    cvtColor(rectIzq, grisIzq, COLOR_BGR2GRAY);
    cvtColor(rectDer, grisDer, COLOR_BGR2GRAY);
    clahe->apply(grisIzq, outIzq);
    clahe->apply(grisDer, outDer);
    GaussianBlur(outIzq, outIzq, Size(3, 3), 0);
    GaussianBlur(outDer, outDer, Size(3, 3), 0);
}

Mat Disparidad::calcular(const Mat& rectIzq, const Mat& rectDer) {
    Mat claheIzq, claheDer;
    preprocesar(rectIzq, rectDer, claheIzq, claheDer);

    //  SGBM (izquierda + derecha) + fusión WLS 
    Mat disparidadIzq, disparidadDer, disparidadFiltrada;
    matcherIzquierdo->compute(claheIzq, claheDer, disparidadIzq);
    matcherDerecho->compute(claheDer, claheIzq, disparidadDer);
    filtroWLS->filter(disparidadIzq, rectIzq, disparidadFiltrada, disparidadDer);
    return disparidadFiltrada;
}

Mat Disparidad::calcularCrudo(const Mat& rectIzq, const Mat& rectDer) {
    Mat claheIzq, claheDer;
    preprocesar(rectIzq, rectDer, claheIzq, claheDer);
    Mat disparidadIzq;
    matcherIzquierdo->compute(claheIzq, claheDer, disparidadIzq);
    return disparidadIzq;
}


//esta de quitar este 
Mat Disparidad::colorizar(const Mat& disparidad16) {
    Mat disparidadFloat, disparidadVis;
    disparidad16.convertTo(disparidadFloat, CV_32F, 1.0 / 16.0);
    Mat pixelesValidos = disparidadFloat > 0;

    double minDisp, maxDisp;
    minMaxLoc(disparidadFloat, &minDisp, &maxDisp, nullptr, nullptr, pixelesValidos);
    if (maxDisp > minDisp) {
        disparidadFloat -= minDisp;
        disparidadFloat *= 255.0 / (maxDisp - minDisp);
    }
    disparidadFloat.convertTo(disparidadVis, CV_8U);
    disparidadVis.setTo(0, ~pixelesValidos);

    Mat mapaDisparidad;
    cvtColor(disparidadVis, mapaDisparidad, COLOR_GRAY2BGR);
    return mapaDisparidad;
}
