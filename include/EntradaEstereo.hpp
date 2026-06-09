#ifndef ENTRADA_ESTEREO_HPP
#define ENTRADA_ESTEREO_HPP

#include <opencv2/core/core.hpp>  // Mat, Size

#include <string>

#include "CameraSource.hpp"


class EntradaEstereo {
public:
    bool iniciar(const std::string& rutaConfig = "config/setup.yml");

    bool leer(cv::Mat& rectIzq, cv::Mat& rectDer);

    const cv::Mat&     getQ()           const { return Q; }              // reproyección 4×4
    double             getDispOffset()  const { return dispOffset; }     // offset de disparidad (px)
    const std::string& getRutaCascara() const { return rutaCascara; }    // cascade facial (efecto AR)

private:
    CameraSource camaraIzquierda, camaraDerecha;
    cv::Mat  mapaIzqX, mapaIzqY, mapaDerX, mapaDerY; 
    cv::Mat  Q;
    cv::Size tamanoImagen;
    double   dispOffset = 0.0;
    std::string rutaCascara;
};

#endif
