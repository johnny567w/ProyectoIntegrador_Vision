#ifndef DISPARIDAD_HPP
#define DISPARIDAD_HPP

#include <opencv2/core/core.hpp>     // Mat
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc.hpp>      // filtro WLS


class Disparidad {
public:
    Disparidad();
    cv::Mat calcular(const cv::Mat& rectIzq, const cv::Mat& rectDer);

    cv::Mat calcularCrudo(const cv::Mat& rectIzq, const cv::Mat& rectDer);

    static cv::Mat colorizar(const cv::Mat& disparidad16);

private:
    void preprocesar(const cv::Mat& rectIzq, const cv::Mat& rectDer,
                     cv::Mat& outIzq, cv::Mat& outDer);

    cv::Ptr<cv::CLAHE>                        clahe;
    cv::Ptr<cv::StereoSGBM>                   matcherIzquierdo;
    cv::Ptr<cv::StereoMatcher>                matcherDerecho;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> filtroWLS;
};

#endif
