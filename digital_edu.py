import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Завантаження даних
data = pd.read_csv('train.csv')

# Функція для розбиття дати народження на рік
def split_bdate(bdate):
    if isinstance(bdate, str) and len(bdate.split('.')) == 3:
        day, month, year = bdate.split('.')
        return int(year)
    elif isinstance(bdate, str) and len(bdate.split('.')) == 2:
        return None  # Відсутній рік
    else:
        return None

data['birth_year'] = data['bdate'].apply(split_bdate)

# Заповнення відсутніх років на основі медіани по кожній статі
def fill_byear(row):
    if pd.isnull(row['birth_year']):
        if row['sex'] == 1:
            return data[data['sex'] == 1]['birth_year'].median()
        else:
            return data[data['sex'] == 2]['birth_year'].median()
    return row['birth_year']

data['birth_year'] = data.apply(fill_byear, axis=1)

# Видалення оригінальної колонки 'bdate'
data.drop('bdate', axis=1, inplace=True)

# Функція для перетворення статі у числовий формат
def convert_sex(sex):
    if sex == 2:
        return 1
    else:
        return 0

data['sex'] = data['sex'].apply(convert_sex)

# Функція для перетворення форми навчання у числовий формат
def convert_education_form(education_form):
    if education_form == 'Full-time':
        return 1
    else:
        return 0

data['education_form'] = data['education_form'].apply(convert_education_form)

# Функція для перевірки наявності російської мови у списку мов
def convert_langs(langs):
    if 'Русский' in str(langs):
        return 1
    else:
        return 0

data['langs'] = data['langs'].apply(convert_langs)

# Перетворення категорійних змінних у числовий формат
categorical_columns = ['education_status', 'relation', 'occupation_type']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Функція для перетворення бінарних значень у числовий формат
def convert_binary(value):
    if str(value).lower() == 'true':
        return 1
    else:
        return 0

binary_columns = ['has_photo', 'has_mobile', 'life_main', 'people_main', 'career_start', 'career_end', 'result']
for col in binary_columns:
    data[col] = data[col].apply(convert_binary)

# Видалення зайвих колонок
data.drop(['id', 'city', 'occupation_name', 'last_seen'], axis=1, inplace=True)

# Заміна пропусків у колонках з числовими значеннями
data.fillna(0, inplace=True)

# Поділ на тренінговий та тестовий набори
X = data.drop('result', axis=1)
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Масштабування даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Побудова моделі k-найближчих сусідів
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Прогнозування та оцінка моделі
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)