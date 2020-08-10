import os
import glob


def rename_files():
    """
    Rename our datasets names and images names for comfort

    Returns:
    sorted_by_class : list of lists sorted by class
    all_img_names : list with all images names """

    all_img_names = []
    folder_names = []

    for x in range(16):
        folder_names = glob.glob("dataset/*")  # store images names in list

    for folder in folder_names:
        os.rename(folder, 'dataset/' + folder[-3:])  # rename datasets folders names to last 3 digit (they are unique)

    folder_names = sorted(folder_names)

    # rename images names to last 7 symbols (they are unique)

    for root, dirs, files in os.walk("dataset"):
        for name in dirs + files:
            img_name = os.path.join(root, name[-7:])
            os.rename(os.path.join(root, name), img_name)
            all_img_names.append(img_name)

    # remove folder names from all_img_names

    all_img_names = list(set(all_img_names) - set(folder_names))
    all_img_names = sorted(all_img_names)

    sorted_by_class = []
    j = 0
    for i in range(len(folder_names)):
        temporary_list = all_img_names[j:j + 100]
        sorted_by_class.append(temporary_list)
        j += 100

    return sorted_by_class, all_img_names, folder_names

