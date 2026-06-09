#include "CameraSource.hpp"
#include <iostream>
#include <cctype>

using namespace cv;
using namespace std;

CameraSource::CameraSource() {}

CameraSource::~CameraSource() {
    activo = false;
    if (hiloCapturaFondo.joinable()) hiloCapturaFondo.join();
}

bool CameraSource::esNumero(const string& texto) const {
    if (texto.empty()) return false;
    for (char c : texto)
        if (!isdigit(c)) return false;
    return true;
}

bool CameraSource::abrir(const string& fuente) {
    if (esNumero(fuente))
        captura.open(stoi(fuente));
    else
        captura.open(fuente);

    if (!captura.isOpened()) {
        cerr << "No se pudo abrir la fuente de video: " << fuente << endl;
        return false;
    }

    captura.set(CAP_PROP_BUFFERSIZE, 1); 

    activo = true;
    hiloCapturaFondo = thread(&CameraSource::bucleCaptura, this);
    return true;
}
void CameraSource::bucleCaptura() {
    Mat fotograma;
    while (activo) {
        if (captura.read(fotograma) && !fotograma.empty()) {
            Reloj::time_point ahora = Reloj::now();
            lock_guard<mutex> lk(mutexFotograma);
            ultimoFotograma = fotograma.clone();
            marcaUltimo     = ahora;
            tieneFotograma  = true;
        } else {
            this_thread::sleep_for(chrono::milliseconds(2));  
        }
    }
}

bool CameraSource::leer(Mat& fotograma) {
    if (!tieneFotograma) return false;
    lock_guard<mutex> lk(mutexFotograma);
    fotograma = ultimoFotograma.clone();
    tieneFotograma = false;  
    return !fotograma.empty();
}

bool CameraSource::leer(Mat& fotograma, Reloj::time_point& marca) {
    if (!tieneFotograma) return false;
    lock_guard<mutex> lk(mutexFotograma);
    fotograma = ultimoFotograma.clone();
    marca     = marcaUltimo;
    tieneFotograma = false;  
    return !fotograma.empty();
}

