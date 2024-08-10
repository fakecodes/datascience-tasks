import numpy as np
from collections import Counter

# Data rumah (lat, long, lokasi)
data = {
    'A': (11, 26, 'Kota'),
    'B': (15, 29, 'Kota'),
    'C': (19, 28, 'Kota'),
    'D': (18, 30, 'Kota'),
    'E': (16, 26, 'Kota'),
    'F': (23, 25, 'Kabupaten'),
    'G': (25, 22, 'Kabupaten'),
    'H': (21, 24, 'Kabupaten'),
    'I': (23, 25, 'Kabupaten'),
    'J': (29, 24, 'Kabupaten'),
}

# Lokasi rumah X
rumah_X = (19, 25)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Hitung jarak rumah X ke setiap rumah dalam data
distances = []
for rumah, (lat, long, lokasi) in data.items():
    dist = euclidean_distance(rumah_X, (lat, long))
    distances.append((dist, lokasi))

# Tentukan nilai k
k = 3

# Ambil k tetangga terdekat
distances.sort(key=lambda x: x[0])
k_nearest_neighbors = distances[:k]

# Tentukan lokasi mayoritas dari k tetangga terdekat
locations = [loc for _, loc in k_nearest_neighbors]
most_common_location = Counter(locations).most_common(1)[0][0]

# Hasil
print(f"Rumah X kemungkinan besar berada di: {most_common_location}")
