CXX = g++

OPENCV_PATH = /home/pepe/opencv1

CXXFLAGS = -std=c++17 -O2 -Wall -Wextra \
	-Iinclude \
	-I$(OPENCV_PATH)/include/opencv4

LIBS = -L$(OPENCV_PATH)/lib \
	-lopencv_core \
	-lopencv_imgproc \
	-lopencv_highgui \
	-lopencv_imgcodecs \
	-lopencv_videoio \
	-lopencv_calib3d \
	-lopencv_ximgproc \
	-lopencv_objdetect \
	-lpthread

# Utilidades base usadas por las herramientas de captura/calibración
SRC_COMMON = src/CameraSource.cpp src/CalibrationUtils.cpp

# Módulos del pipeline estéreo, separados por concern:
#   EntradaEstereo -> E/S (config + calib + cámaras sincronizadas y rectificadas)
#   Disparidad     -> SGBM + WLS
#   Profundidad    -> clases Distancia (reproyección Q) + Filtro (mediana + Kalman)
#   AREffect       -> efecto de realidad aumentada "Big Red Head"
SRC_ENTRADA    = src/EntradaEstereo.cpp src/CameraSource.cpp
SRC_DISPARIDAD = src/Disparidad.cpp
SRC_PROFUNDIDAD = src/Profundidad.cpp
SRC_EFECTO      = src/AREffect.cpp

BIN_DIR = bin

all: dirs capture_pairs stereo_calibrate rectify_preview disparidad distancia efecto sin_correccion

dirs:
	mkdir -p $(BIN_DIR)
	mkdir -p data/calibration/left
	mkdir -p data/calibration/right
	mkdir -p config
	mkdir -p build

capture_pairs: apps/capture_pairs.cpp $(SRC_COMMON)
	$(CXX) $(CXXFLAGS) apps/capture_pairs.cpp $(SRC_COMMON) -o $(BIN_DIR)/capture_pairs $(LIBS)

stereo_calibrate: apps/stereo_calibrate.cpp $(SRC_COMMON)
	$(CXX) $(CXXFLAGS) apps/stereo_calibrate.cpp $(SRC_COMMON) -o $(BIN_DIR)/stereo_calibrate $(LIBS)

rectify_preview: apps/rectify_preview.cpp $(SRC_COMMON)
	$(CXX) $(CXXFLAGS) apps/rectify_preview.cpp $(SRC_COMMON) -o $(BIN_DIR)/rectify_preview $(LIBS)

# Ejecutable de DISPARIDAD: sólo muestra el mapa de disparidad
disparidad: apps/disparidad.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD)
	$(CXX) $(CXXFLAGS) apps/disparidad.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD) -o $(BIN_DIR)/disparidad $(LIBS)

# Ejecutable de DISTANCIA: ventana combinada distancia (cruda) + filtro (Kalman)
distancia: apps/distancia.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD) $(SRC_PROFUNDIDAD)
	$(CXX) $(CXXFLAGS) apps/distancia.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD) $(SRC_PROFUNDIDAD) -o $(BIN_DIR)/distancia $(LIBS)

# Ejecutable del EFECTO AR: ventana individual "Big Red Head" (usa el pipeline completo)
efecto: apps/efecto.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD) $(SRC_PROFUNDIDAD) $(SRC_EFECTO)
	$(CXX) $(CXXFLAGS) apps/efecto.cpp $(SRC_ENTRADA) $(SRC_DISPARIDAD) $(SRC_PROFUNDIDAD) $(SRC_EFECTO) -o $(BIN_DIR)/efecto $(LIBS)

sin_correccion: apps/sin_correccion.cpp $(SRC_COMMON)
	$(CXX) $(CXXFLAGS) apps/sin_correccion.cpp $(SRC_COMMON) -o $(BIN_DIR)/sin_correccion $(LIBS)

run_capture:
	./$(BIN_DIR)/capture_pairs

run_calibrate:
	./$(BIN_DIR)/stereo_calibrate

run_rectify:
	./$(BIN_DIR)/rectify_preview

run_disparidad:
	./$(BIN_DIR)/disparidad

run_distancia:
	./$(BIN_DIR)/distancia

run_efecto:
	./$(BIN_DIR)/efecto

clean:
	rm -rf $(BIN_DIR) build
