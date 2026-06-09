#ifndef AR_EFFECT_HPP
#define AR_EFFECT_HPP

#include <opencv2/core/core.hpp>        // Mat — definiciones base de matrices
#include <opencv2/imgproc/imgproc.hpp>  // Procesamiento de imágenes
#include <opencv2/highgui/highgui.hpp>  // Creación de ventanas
#include <opencv2/objdetect.hpp>        // CascadeClassifier — detección de objetos

#include <string>

bool cargarCascara(cv::CascadeClassifier& cascara, const std::string& ruta);


void aplicarCabezaRoja(cv::Mat& fotograma, float distanciaCm,
                       cv::CascadeClassifier& cascara);

#endif
