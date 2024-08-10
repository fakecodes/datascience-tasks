import pandas as pd

# Data karyawan
data = {
    'Department': ['Sales', 'Sales', 'Sales', 'Systems', 'Systems', 'Systems', 'Systems', 'Marketing', 'Marketing', 'Secretary', 'Secretary'],
    'Status': ['Senior', 'Junior', 'Junior', 'Junior', 'Senior', 'Junior', 'Senior', 'Senior', 'Junior', 'Senior', 'Junior'],
    'Usia': ['31-35 thn', '26-30 thn', '31-35 thn', '21-25 thn', '31-35 thn', '26-30 thn', '41-45 thn', '36-40 thn', '31-35 thn', '46-50 thn', '26-30 thn'],
    'Gaji': ['46-50 jt', '26-30 jt', '31-35 jt', '46-50 jt', '66-70 jt', '46-50 jt', '66-70 jt', '46-50 jt', '41-45 jt', '36-40 jt', '36-40 jt'],
    'Jumlah': [30, 40, 40, 20, 5, 3, 3, 10, 4, 4, 6]
}

# Buat dataframe
df = pd.DataFrame(data)

# Hitung Probabilitas Prior
total_karyawan = df['Jumlah'].sum()
prob_senior = df[df['Status'] == 'Senior']['Jumlah'].sum() / total_karyawan
prob_junior = df[df['Status'] == 'Junior']['Jumlah'].sum() / total_karyawan

# Fungsi untuk menghitung likelihood
def likelihood(df, attr, value, status):
    subset = df[df['Status'] == status]
    count = subset[subset[attr] == value]['Jumlah'].sum()
    total = subset['Jumlah'].sum()
    return count / total if total > 0 else 0

# Data X
X = {'Department': 'Systems', 'Usia': '26-30 thn', 'Gaji': '46-50 jt'}

# Hitung Likelihood untuk Senior
likelihood_senior = (
    likelihood(df, 'Department', X['Department'], 'Senior') *
    likelihood(df, 'Usia', X['Usia'], 'Senior') *
    likelihood(df, 'Gaji', X['Gaji'], 'Senior')
)

# Hitung Likelihood untuk Junior
likelihood_junior = (
    likelihood(df, 'Department', X['Department'], 'Junior') *
    likelihood(df, 'Usia', X['Usia'], 'Junior') *
    likelihood(df, 'Gaji', X['Gaji'], 'Junior')
)

# Hitung Probabilitas Posterior
posterior_senior = prob_senior * likelihood_senior
posterior_junior = prob_junior * likelihood_junior

# Tentukan class label berdasarkan probabilitas posterior tertinggi
class_label = 'Senior' if posterior_senior > posterior_junior else 'Junior'

print(f"Status/class label dari data/tuple X tersebut adalah: {class_label}")
