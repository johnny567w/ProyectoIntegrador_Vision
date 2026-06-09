#ifndef CAMERA_SOURCE_HPP
#define CAMERA_SOURCE_HPP

#include <opencv2/videoio/videoio.hpp>  // VideoCapture — lectura de video y streaming
#include <opencv2/core/core.hpp>        // Mat — definiciones base de matrices

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

class CameraSource {
public:
    using Reloj = std::chrono::steady_clock;

    CameraSource();
    ~CameraSource();

    bool abrir(const std::string& fuente);  // Abre cámara por índice numérico o URL
    bool leer(cv::Mat& fotograma);          // Copia el último fotograma capturado
    bool leer(cv::Mat& fotograma, Reloj::time_point& marca);

private:
    cv::VideoCapture    captura;
    cv::Mat             ultimoFotograma;
    Reloj::time_point   marcaUltimo;        
    std::mutex          mutexFotograma;
    std::thread         hiloCapturaFondo;
    std::atomic<bool>   activo{false};
    std::atomic<bool>   tieneFotograma{false};

    void bucleCaptura();
    bool esNumero(const std::string& texto) const;
};

#endif
