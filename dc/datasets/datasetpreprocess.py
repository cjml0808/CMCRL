import os


def makeimagestxt():
    folder1_path = 'E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/images'
    folder1_list = os.listdir(folder1_path)
    image_list = []
    num = 0
    for i in range(len(folder1_list)):
        sub_path = os.path.join(folder1_path, folder1_list[i])
        sub_list = os.listdir(sub_path)
        for j in range(len(sub_list)):
            num += 1
            image_list.append(str(num) + ' ' + os.path.join(folder1_list[i], sub_list[j]) + '\n')

    with open("E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/images.txt", "w") as f:
        f.writelines(image_list)

def makeimage_class_labelstxt():
    folder1_path = 'E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/images'
    folder1_list = os.listdir(folder1_path)
    image_list = []
    num = 0
    for i in range(len(folder1_list)):
        sub_path = os.path.join(folder1_path, folder1_list[i])
        sub_list = os.listdir(sub_path)
        for j in range(len(sub_list)):
            num += 1
            image_list.append(str(num) + ' ' + sub_list[j].split('_')[0] + '\n')

    with open("E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/image_class_labels.txt", "w") as f:
        f.writelines(image_list)

def maketrain_test_splitstxt(ratio=0.8):
    folder1_path = 'E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/images'
    folder1_list = os.listdir(folder1_path)
    image_list = []
    num = 0
    for i in range(len(folder1_list)):
        sub_path = os.path.join(folder1_path, folder1_list[i])
        sub_list = os.listdir(sub_path)
        for j in range(len(sub_list)):
            num += 1
            if j < int(len(sub_list) * ratio):
                image_list.append(str(num) + ' 1' + '\n')
            else:
                image_list.append(str(num) + ' 0' + '\n')

    with open("E:/Code/densityclustering-master/examples/data/citrusdisease6/citrus_disease_6/train_test_split.txt", "w") as f:
        f.writelines(image_list)



if __name__ == '__main__':
    # makeimagestxt()
    # makeimage_class_labelstxt()
    # maketrain_test_splitstxt(ratio=0.8)
    maketrain_test_splitstxt(ratio=0.5)