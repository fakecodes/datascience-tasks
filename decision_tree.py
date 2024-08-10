import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

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

# Encoding atribut kategorikal
df_encoded = pd.get_dummies(df.drop('Status', axis=1))
labels = df['Status']

# Buat decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(df_encoded, labels)

# Tampilkan decision tree dalam bentuk teks
tree_rules = export_text(clf, feature_names=list(df_encoded.columns))
print(tree_rules)
