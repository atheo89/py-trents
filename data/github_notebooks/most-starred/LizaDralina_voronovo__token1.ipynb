import vk_api
import csv
import pandas as pd


vk_token = 'vk1.a. ' # Токен доступа к API ВКонтакте
vk_session = vk_api.VkApi(token=vk_token) # Создание сессии ВКонтакте с использованием токена

vk = vk_session.get_api() # Получение доступа к методам API ВКонтакте

v = vk.wall.get(count = 1)['items'][0]['copy_history'][0]['text'] # Получение текста последней записи со стены

# Поиск пользователей в указанном городе (в данном случае во Владивостоке),
# получение информации о них и сохранение в переменную users
users = vk.users.search(count=1000, city_id=37, fields='bdate, city, home_town, nickname, sex, verified', sort = 1)['items']

# Проход по всем найденным пользователям
for one in users:
   if 'bdate' in one: # Проверка, есть ли у пользователя дата рождения
    date_b = one['bdate']
    date_b = date_b.split(".") # Разделение строки с датой рождения на компоненты
    # Проверка, содержит ли строка три компонента (день, месяц, год)
    if len(date_b) == 3:
        year = date_b[2] # Извлечение года рождения
        # Добавление года рождения к данным пользователя
        one['year'] = year

# Создание DataFrame из данных о пользователях
data = pd.DataFrame(users)
data.to_csv('voronovo.csv', index=False)