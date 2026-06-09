#ifndef CALIBRATION_UTILS_HPP
#define CALIBRATION_UTILS_HPP

#include <opencv2/core/core.hpp>        // Mat, Size, Point3f — definiciones base
#include <opencv2/imgproc/imgproc.hpp>  // Procesamiento de imágenes

#include <string>
#include <vector>

std::vector<cv::Point3f> crearPuntosObjeto(cv::Size tamanoTablero, float tamCuadroMm);

std::vector<std::string> obtenerImagenes(const std::string& carpeta);

void dibujarGuiasHorizontales(cv::Mat& imagen, int espaciado);


#endif
