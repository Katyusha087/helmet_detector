import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Путь к папке с файлами аннотаций
annotations = "SafetyHelmetDetection/annotations"
output_dir = "output/graficks"
# Словарь для хранения количества объектов каждого класса
class_counts = {}
# Словари для хранения количества файлов с шлемами и головами
helmet_counts = {'only_helmet': 0, 'helmet_and_head': 0}
head_counts = {'only_head': 0, 'helmet_and_head': 0}

# Пройтись по всем файлам аннотаций в директории
for annotation_file in glob.glob(os.path.join(annotations, '*.xml')):
    # Прочитать файл аннотации
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # Флаги, показывающие, есть ли в файле шлемы или головы
    has_helmet = False
    has_head = False

    # Пройтись по каждому объекту в файле аннотации
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name == 'helmet':
            has_helmet = True
        elif class_name == 'head':
            has_head = True
        elif class_name == 'person':
            continue
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Увеличить счетчик файлов с шлемом или головой, если они есть
    if has_helmet and not has_head:
        helmet_counts['only_helmet'] += 1
    elif has_head and not has_helmet:
        head_counts['only_head'] += 1
    elif has_helmet and has_head:
        helmet_counts['helmet_and_head'] += 1
        head_counts['helmet_and_head'] += 1

# Отобразить результаты в виде круговых диаграмм
# Диаграмма распределения классов
labels = class_counts.keys()
sizes = class_counts.values()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Распределение классов объектов')

# Сохранить график в файл
plt.savefig(output_dir + "/" + 'class_distribution.png')

# Диаграмма распределения файлов с шлемами и головами
labels = ['Только шлемы', 'Шлемы и головы', 'Только головы']
sizes = [helmet_counts['only_helmet'], helmet_counts['helmet_and_head'], head_counts['only_head']]
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Распределение файлов с шлемами и головами')

# Сохранить график в файл
plt.savefig(output_dir + "/" + 'helmet_head_distribution.png')
