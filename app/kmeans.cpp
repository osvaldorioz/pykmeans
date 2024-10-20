#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) kmeans.cpp -o kmeans_module$(python3.12-config --extension-suffix)

namespace py = pybind11;

class KMeans {
public:
    KMeans(int k, int max_iters)
        : k(k), max_iters(max_iters) {}

    // Método para ajustar el modelo de k-means
    void fit(py::array_t<double> data) {
        auto buf = data.request();
        double* data_ptr = static_cast<double*>(buf.ptr);

        int n_samples = buf.shape[0];
        int n_features = buf.shape[1];

        // Inicializar centroides de forma aleatoria
        initialize_centroids(data_ptr, n_samples, n_features);

        // Realizar iteraciones
        for (int iter = 0; iter < max_iters; ++iter) {
            // Asignar cada punto al centroide más cercano
            std::vector<int> labels(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                labels[i] = closest_centroid(&data_ptr[i * n_features], n_features);
            }

            // Actualizar los centroides
            update_centroids(data_ptr, labels, n_samples, n_features);
        }
    }

    // Método para predecir el centroide más cercano para nuevos datos
    py::array_t<int> predict(py::array_t<double> data) {
        auto buf = data.request();
        double* data_ptr = static_cast<double*>(buf.ptr);

        int n_samples = buf.shape[0];
        int n_features = buf.shape[1];

        py::array_t<int> labels({n_samples});
        auto labels_buf = labels.request();
        int* labels_ptr = static_cast<int*>(labels_buf.ptr);

        for (int i = 0; i < n_samples; ++i) {
            labels_ptr[i] = closest_centroid(&data_ptr[i * n_features], n_features);
        }

        return labels;
    }

    // Obtener los centroides finales
    py::array_t<double> get_centroids() {
        py::array_t<double> centroids({k, n_features});
        auto buf = centroids.request();
        double* centroids_ptr = static_cast<double*>(buf.ptr);
        std::copy(centroids_.begin(), centroids_.end(), centroids_ptr);
        return centroids;
    }

private:
    int k;  // Número de clusters
    int max_iters;  // Iteraciones máximas
    int n_features;  // Número de características (dimensionalidad de los datos)
    std::vector<double> centroids_;  // Centroides de los clusters

    // Inicializar centroides aleatoriamente
    void initialize_centroids(double* data, int n_samples, int n_features) {
        this->n_features = n_features;
        centroids_.resize(k * n_features);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n_samples - 1);

        for (int i = 0; i < k; ++i) {
            int random_index = dis(gen);
            for (int j = 0; j < n_features; ++j) {
                centroids_[i * n_features + j] = data[random_index * n_features + j];
            }
        }
    }

    // Encontrar el centroide más cercano a un punto
    int closest_centroid(double* point, int n_features) {
        int closest = 0;
        double min_dist = std::numeric_limits<double>::max();

        for (int i = 0; i < k; ++i) {
            double dist = 0.0;
            for (int j = 0; j < n_features; ++j) {
                double diff = point[j] - centroids_[i * n_features + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                closest = i;
            }
        }

        return closest;
    }

    // Actualizar los centroides basándose en las asignaciones de los puntos
    void update_centroids(double* data, const std::vector<int>& labels, int n_samples, int n_features) {
        std::vector<int> count(k, 0);
        std::fill(centroids_.begin(), centroids_.end(), 0.0);

        for (int i = 0; i < n_samples; ++i) {
            int cluster_id = labels[i];
            count[cluster_id]++;
            for (int j = 0; j < n_features; ++j) {
                centroids_[cluster_id * n_features + j] += data[i * n_features + j];
            }
        }

        for (int i = 0; i < k; ++i) {
            if (count[i] == 0) continue;
            for (int j = 0; j < n_features; ++j) {
                centroids_[i * n_features + j] /= count[i];
            }
        }
    }
};

// Enlace con Pybind11
PYBIND11_MODULE(kmeans_module, m) {
    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int, int>())
        .def("fit", &KMeans::fit)
        .def("predict", &KMeans::predict)
        .def("get_centroids", &KMeans::get_centroids);
}
