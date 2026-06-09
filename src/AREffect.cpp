#include "AREffect.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

static Vec3b paletaFuego(float brillo) {
    // se define el color que tendra cada etapa del filtro segun que tan cerca o lejos se este
    if (brillo < 64.0f) {
        float t = brillo / 64.0f;
        return Vec3b(0, 0, (uchar)(150.0f * t));                          // negro → rojo oscuro
    } else if (brillo < 128.0f) {
        float t = (brillo - 64.0f) / 64.0f;
        return Vec3b(0, (uchar)(100.0f * t), (uchar)(150.0f + 105.0f * t)); // rojo → naranja
    } else if (brillo < 192.0f) {
        float t = (brillo - 128.0f) / 64.0f;
        return Vec3b(0, (uchar)(100.0f + 155.0f * t), 255);               // naranja → amarillo
    } else {
        float t = (brillo - 192.0f) / 63.0f;
        return Vec3b((uchar)(255.0f * t), 255, 255);                      // amarillo → blanco
    }
}

//carga haar
bool cargarCascara(CascadeClassifier& cascara, const string& ruta) {
    if (!cascara.load(ruta)) {
        cerr << "No se pudo cargar la cascara: " << ruta << endl;
        return false;
    }
    cout << "Cascara facial cargada OK" << endl;
    return true;
}


void aplicarCabezaRoja(Mat& fotograma, float distanciaCm,
                       CascadeClassifier& cascara) {

    // ── 1. Detección de caras 
    static Rect  cajaCaraDetectada;
    static bool  tieneCara          = false;
    static int   contadorDeteccion  = 0;

    if (contadorDeteccion % 4 == 0) {
        Mat gris, reducida;
        cvtColor(fotograma, gris, COLOR_BGR2GRAY);
        resize(gris, reducida, Size(), 0.5, 0.5);
        equalizeHist(reducida, reducida);

        vector<Rect> carasDetectadas;
        cascara.detectMultiScale(reducida, carasDetectadas, 1.1, 4, 0,
                                 Size(20, 20), Size(reducida.cols, reducida.rows));

        tieneCara = !carasDetectadas.empty();
        if (tieneCara) {
            Rect caraPrincipal = *max_element(carasDetectadas.begin(), carasDetectadas.end(),
                [](const Rect& a, const Rect& b){ return a.area() < b.area(); });
            cajaCaraDetectada = Rect(caraPrincipal.x * 2, caraPrincipal.y * 2,
                                    caraPrincipal.width * 2, caraPrincipal.height * 2);
        }
    }
    contadorDeteccion++;

    if (!tieneCara) return;

//saber que tan lejos se enuentra, segun eso va creciendo
    const float DISTANCIA_LEJOS  = 200.0f;
    const float DISTANCIA_CERCA  =  30.0f;

    float t = clamp((distanciaCm - DISTANCIA_CERCA) / (DISTANCIA_LEJOS - DISTANCIA_CERCA),
                    0.0f, 1.0f);
    float escalaObjetivo = 3.0f - 2.0f * t;  

    static float escalaActual = 1.0f;
    escalaActual += (escalaObjetivo - escalaActual) * 0.03f;

    Point2f centroCara(cajaCaraDetectada.x + cajaCaraDetectada.width  * 0.5f,
                       cajaCaraDetectada.y + cajaCaraDetectada.height * 0.5f);

    int margen = (int)(cajaCaraDetectada.height * 0.5f);
    int rx0 = max(0, cajaCaraDetectada.x - margen);
    int ry0 = max(0, cajaCaraDetectada.y - margen);
    int rx1 = min(fotograma.cols, cajaCaraDetectada.x + cajaCaraDetectada.width  + margen);
    int ry1 = min(fotograma.rows, cajaCaraDetectada.y + cajaCaraDetectada.height + margen);
    Rect regionCabeza(rx0, ry0, rx1 - rx0, ry1 - ry0);
    if (regionCabeza.width <= 0 || regionCabeza.height <= 0) return;

    //  3. Extraer región de la cabeza y agrandarla con resize 
    Mat cabezaOriginal = fotograma(regionCabeza).clone();
    int nuevoAncho = (int)(regionCabeza.width  * escalaActual);
    int nuevoAlto  = (int)(regionCabeza.height * escalaActual);
    if (nuevoAncho <= 0 || nuevoAlto <= 0) return;

    Mat cabezaAmpliada;
    resize(cabezaOriginal, cabezaAmpliada, Size(nuevoAncho, nuevoAlto), 0, 0, INTER_LINEAR);

    //  4.camabiar el color pixel a pixel 
    float rojoBase = 1.0f - t;
    float pulso    = 0.08f * sin((float)getTickCount() /
                                 (float)getTickFrequency() * 4.0f * (float)M_PI);
    float factorFuego = clamp(rojoBase + pulso, 0.0f, 1.0f);

    Vec3b pixel;
    for (int i = 0; i < cabezaAmpliada.rows; i++) {
        for (int j = 0; j < cabezaAmpliada.cols; j++) {
            pixel = cabezaAmpliada.at<Vec3b>(i, j);

            float brillo = 0.114f * pixel[0] + 0.587f * pixel[1] + 0.299f * pixel[2];

            Vec3b colorFuego = paletaFuego(brillo);

            pixel[0] = (uchar)(pixel[0] * (1.0f - factorFuego) + colorFuego[0] * factorFuego);
            pixel[1] = (uchar)(pixel[1] * (1.0f - factorFuego) + colorFuego[1] * factorFuego);
            pixel[2] = (uchar)(pixel[2] * (1.0f - factorFuego) + colorFuego[2] * factorFuego);

            cabezaAmpliada.at<Vec3b>(i, j) = pixel;
        }
    }

    Mat mascara = Mat::zeros(nuevoAlto, nuevoAncho, CV_32F);
    ellipse(mascara, Point(nuevoAncho / 2, nuevoAlto / 2),
            Size(nuevoAncho / 2 - 2, nuevoAlto / 2 - 2),
            0, 0, 360, Scalar(1.0f), -1);
    int tamanoDesenfoque = min((int)(nuevoAncho * 0.12f) | 1, 21);
    GaussianBlur(mascara, mascara, Size(tamanoDesenfoque, tamanoDesenfoque), 0);

    //  6. Halo naranja detrás de la cabeza 
    if (factorFuego > 0.05f) {
//cambia entre amarillo y naranja
            float tiempoHalo = (float)getTickCount() / (float)getTickFrequency();
        float pulsoHalo  = 0.5f + 0.5f * sin(tiempoHalo * 3.0f);
        // Color base del halo: naranja (0, 128, 255) → amarillo (0, 255, 255)
        uchar haloG = (uchar)(128.0f + 127.0f * pulsoHalo);
        Scalar colorHalo(0, haloG, 255);

//hacer mas grande el halo segun el zoom de arriba

        int radioX = (int)(nuevoAncho * 0.6f);
        int radioY = (int)(nuevoAlto  * 0.6f);

        Mat capaHalo = Mat::zeros(fotograma.rows, fotograma.cols, CV_8UC3);
        ellipse(capaHalo,
                Point((int)centroCara.x, (int)centroCara.y),
                Size(radioX, radioY),
                0, 0, 360, colorHalo, -1);

        int tamBlurHalo = 31;
        GaussianBlur(capaHalo, capaHalo, Size(tamBlurHalo, tamBlurHalo), 0);

        addWeighted(fotograma, 1.0f, capaHalo, factorFuego * 0.45f, 0.0, fotograma);
    }

    //  7. Pegar cabeza ampliada 
    int destX = (int)centroCara.x - nuevoAncho / 2;
    int destY = (int)centroCara.y - nuevoAlto  / 2;

    int desplazX = max(0, -destX);
    int desplazY = max(0, -destY);
    destX = max(0, destX);
    destY = max(0, destY);

    int anchoCopiado = min(nuevoAncho - desplazX, fotograma.cols - destX);
    int altoCopiado  = min(nuevoAlto  - desplazY, fotograma.rows - destY);
    if (anchoCopiado <= 0 || altoCopiado <= 0) return;

    Vec3b pixelFuente, pixelDestino;
    float alpha;
    for (int i = 0; i < altoCopiado; i++) {
        for (int j = 0; j < anchoCopiado; j++) {
            pixelFuente  = cabezaAmpliada.at<Vec3b>(desplazY + i, desplazX + j);
            pixelDestino = fotograma.at<Vec3b>(destY  + i, destX  + j);
            alpha        = mascara.at<float>(desplazY + i, desplazX + j);

            fotograma.at<Vec3b>(destY + i, destX + j) = Vec3b(
                (uchar)(pixelFuente[0] * alpha + pixelDestino[0] * (1.0f - alpha)),
                (uchar)(pixelFuente[1] * alpha + pixelDestino[1] * (1.0f - alpha)),
                (uchar)(pixelFuente[2] * alpha + pixelDestino[2] * (1.0f - alpha))
            );
        }
    }

    // efecto rojo alrededor del amarillo o naranja
    if (factorFuego > 0.05f) {
        float tiempoSeg = (float)getTickCount() / (float)getTickFrequency();
        int altoLlama   = (int)(nuevoAlto * 0.45f * factorFuego);  

        int llX0 = destX;
        int llX1 = destX + anchoCopiado;
        int llY1 = destY;              

        for (int j = llX0; j < llX1; j++) {
            float fase    = (float)(j - llX0) / (float)max(anchoCopiado - 1, 1);
            float ondeo   = 0.5f + 0.5f * sin(fase * 6.0f + tiempoSeg * 8.0f)
                                 * sin(fase * 3.2f + tiempoSeg * 5.3f)
                                 * sin(fase * 1.7f + tiempoSeg * 3.1f);
            int columnaAlto = (int)(altoLlama * ondeo);

            for (int i = llY1 - columnaAlto; i < llY1; i++) {
                if (i < 0 || i >= fotograma.rows || j < 0 || j >= fotograma.cols) continue;

                float posRel = (float)(i - (llY1 - columnaAlto)) / (float)max(columnaAlto - 1, 1);
                float brillo = posRel * 255.0f;

                Vec3b colorLlama = paletaFuego(brillo);

                float alphaLlama = factorFuego * posRel;

                pixelDestino = fotograma.at<Vec3b>(i, j);
                fotograma.at<Vec3b>(i, j) = Vec3b(
                    (uchar)(colorLlama[0] * alphaLlama + pixelDestino[0] * (1.0f - alphaLlama)),
                    (uchar)(colorLlama[1] * alphaLlama + pixelDestino[1] * (1.0f - alphaLlama)),
                    (uchar)(colorLlama[2] * alphaLlama + pixelDestino[2] * (1.0f - alphaLlama))
                );
            }
        }
    }

}
