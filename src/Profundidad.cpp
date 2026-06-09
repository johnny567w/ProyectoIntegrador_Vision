#include "Profundidad.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;


Distancia::Distancia(const Mat& Qmat) {
    Qmat.convertTo(Q, CV_64F);  
}

double Distancia::reproyectarZ(double x, double y, double disp) const {
    double Z = Q.at<double>(2,0)*x + Q.at<double>(2,1)*y + Q.at<double>(2,2)*disp + Q.at<double>(2,3);
    double W = Q.at<double>(3,0)*x + Q.at<double>(3,1)*y + Q.at<double>(3,2)*disp + Q.at<double>(3,3);
    if (fabs(W) < 1e-9) return 0.0;
    return (Z / W) / 10.0;  
}

double Distancia::getFocalPx() const {
    return Q.at<double>(2,3);            
}

double Distancia::getBaselineCm() const {
    double q32 = Q.at<double>(3,2);      
    if (fabs(q32) < 1e-9) return 0.0;
    return fabs(1.0 / q32) / 10.0;       
}
double Distancia::muestrearDisparidadMediana(const Mat& disp16) const {
    int cx   = disp16.cols / 2;
    int cy   = disp16.rows / 2;
    int half = TAM_PARCHE / 2;

    vector<double> valores;
    valores.reserve(TAM_PARCHE * TAM_PARCHE);

    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            int px = cx + j;
            int py = cy + i;
            if (px < 0 || py < 0 || px >= disp16.cols || py >= disp16.rows) continue;

            double d = disp16.at<short>(py, px) / 16.0;
            if (d <= 0.0) continue;
            valores.push_back(d);
        }
    }

    if (valores.empty()) return 0.0;

    size_t mitad = valores.size() / 2;
    nth_element(valores.begin(), valores.begin() + mitad, valores.end());
    return valores[mitad];
}

double Distancia::calcular(const Mat& disparidad16) const {
    double disp = muestrearDisparidadMediana(disparidad16) - dispOffset;
    if (disp <= 0.0) return 0.0;

    double cx = disparidad16.cols / 2.0;
    double cy = disparidad16.rows / 2.0;
    double distCruda = reproyectarZ(cx, cy, disp);

    if (distCruda < 3.0 || distCruda > 300.0) return 0.0;  // fuera de rango
    return distCruda;
}


double FiltroKalman1D::actualizar(double medicion) {
    double x_pred   = x + v;                
    double Pxx_pred = Pxx + 2.0*Pxv + Pvv;  
    double Pxv_pred = Pxv + Pvv;
    double Pvv_pred = Pvv;

    double innovacion = medicion - x_pred;
    double Q_efectivo = Q + 0.005 * innovacion * innovacion;

    Pxx_pred += 0.25 * Q_efectivo;
    Pxv_pred += 0.50 * Q_efectivo;
    Pvv_pred += Q_efectivo;

    double S  = Pxx_pred + R;   
    double Kx = Pxx_pred / S;   
    double Kv = Pxv_pred / S;   

    x    = x_pred + Kx * innovacion;
    v   += Kv * innovacion;
    Pxx  = (1.0 - Kx) * Pxx_pred;  
    Pxv  = (1.0 - Kx) * Pxv_pred;
    Pvv  = Pvv_pred - Kv * Pxv_pred;

    return x;
}

double Filtro::promedioMovil() const {
    if (bufferPromedio.empty()) return 0.0;
    return accumulate(bufferPromedio.begin(), bufferPromedio.end(), 0.0) / bufferPromedio.size();
}

double Filtro::desviacionEstandar() const {
    if (bufferDesviacion.size() < 2) return 0.0;
    double n     = (double)bufferDesviacion.size();
    double media = accumulate(bufferDesviacion.begin(), bufferDesviacion.end(), 0.0) / n;
    double var   = 0.0;
    for (double val : bufferDesviacion) { double d = val - media; var += d * d; }
    return sqrt(var / (n - 1.0));
}

Filtro::Resultado Filtro::actualizar(double distCruda) {
    Resultado resultado;

    bufferMediana.push_back(distCruda);
    if (bufferMediana.size() > TAM_MEDIANA) bufferMediana.pop_front();

    vector<double> vecMediana(bufferMediana.begin(), bufferMediana.end());
    size_t mitad = vecMediana.size() / 2;
    nth_element(vecMediana.begin(), vecMediana.begin() + mitad, vecMediana.end());
    double distMediana = vecMediana[mitad];

    bufferPromedio.push_back(distMediana);
    if (bufferPromedio.size() > TAM_PROMEDIO) bufferPromedio.pop_front();
    double distPromedio = promedioMovil();

    if (filtroKalman.x == 0.0 || abs(distPromedio - filtroKalman.x) > 60.0) {
        filtroKalman.reiniciar();
        filtroKalman.x = distPromedio;
    }
    double distSuavizada = filtroKalman.actualizar(distPromedio);

    resultado.distFiltrada = distSuavizada;

    bufferDesviacion.push_back(distSuavizada);
    if (bufferDesviacion.size() > TAM_DESVIACION) bufferDesviacion.pop_front();

    resultado.desviacion = desviacionEstandar();
    resultado.estable    = (resultado.desviacion < 5.0);

    return resultado;
}
