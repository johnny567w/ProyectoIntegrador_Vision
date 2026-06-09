#ifndef PROFUNDIDAD_HPP
#define PROFUNDIDAD_HPP

#include <opencv2/core/core.hpp>  // Mat

#include <deque>


class Distancia {
public:
    explicit Distancia(const cv::Mat& Qmat);

    void setDispOffset(double d) { dispOffset = d; }   // offset de disparidad a restar (px)

    double calcular(const cv::Mat& disparidad16) const;

    double getFocalPx()    const;  
    double getBaselineCm() const; 
    static const int TAM_PARCHE = 15; 

    cv::Mat Q;             
    double  dispOffset  = 0.0; 

    double reproyectarZ(double x, double y, double disp) const;
    double muestrearDisparidadMediana(const cv::Mat& disp16) const;
};

struct FiltroKalman1D {
    double x   = 0.0;   // estimado de posición (cm)
    double v   = 0.0;   // estimado de velocidad (cm/cuadro)
    double Pxx = 1.0;   // varianza de posición
    double Pxv = 0.0;   // covarianza cruzada posición-velocidad
    double Pvv = 1.0;   // varianza de velocidad
    double Q;           // ruido de proceso (cm²/cuadro²)
    double R;           // ruido de medición (cm²)

    FiltroKalman1D(double q = 10.0, double r = 5.0) : Q(q), R(r) {}

    double actualizar(double medicion);
    void   reiniciar() { x = 0.0; v = 0.0; Pxx = 1.0; Pxv = 0.0; Pvv = 1.0; }
};

class Filtro {
public:
    struct Resultado {
        double distFiltrada = 0.0;  // distancia suavizada (cm)
        double desviacion   = 0.0;  // desviación estándar de las últimas mediciones
        bool   estable      = false;// true si la medición es estable (desviacion < 5 cm)
    };

    void setKalmanQ(double q) { filtroKalman.Q = q; }
    void setKalmanR(double r) { filtroKalman.R = r; }

    Resultado actualizar(double distCruda);

private:
    FiltroKalman1D filtroKalman;

    static const int TAM_MEDIANA    = 7;   // mediana temporal: elimina spikes de SGBM
    static const int TAM_PROMEDIO   = 5;   // ventana del promedio móvil
    static const int TAM_DESVIACION = 20;  // ventana para desviación estándar

    std::deque<double> bufferMediana;
    std::deque<double> bufferPromedio;
    std::deque<double> bufferDesviacion;

    double promedioMovil()      const;
    double desviacionEstandar() const;
};

#endif
