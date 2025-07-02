# encoding: utf-8

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        num_pids = data["class_num"]
        num_imgs = data["data"].shape[0]
        return num_pids, num_imgs

    def print_dataset_statistics(self, train, test):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, test):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_test_pids, num_test_imgs = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} ".format(num_train_pids, num_train_imgs))
        print("  test     | {:5d} | {:8d} ".format(num_test_pids, num_test_imgs))
        print("  ----------------------------------------")

        return num_train_pids, num_train_imgs, num_test_pids, num_test_imgs



# class BaseDataset(object):
#     """
#     Base class of reid dataset
#     """
#
#     def get_imagedata_info(self, data):
#         pids = []
#         for _, pid in data:
#             pids += [pid]
#         pids = set(pids)
#         num_pids = len(pids)
#         num_imgs = len(data)
#         return num_pids, num_imgs
#
#     def print_dataset_statistics(self, train, test):
#         raise NotImplementedError
#
#     @property
#     def images_dir(self):
#         return None
#
#
# class BaseImageDataset(BaseDataset):
#     """
#     Base class of image reid dataset
#     """
#
#     def print_dataset_statistics(self, train, test):
#         num_train_pids, num_train_imgs = self.get_imagedata_info(train)
#         num_test_pids, num_test_imgs = self.get_imagedata_info(test)
#
#         print("Dataset statistics:")
#         print("  ----------------------------------------")
#         print("  subset   | # ids | # images")
#         print("  ----------------------------------------")
#         print("  train    | {:5d} | {:8d} ".format(num_train_pids, num_train_imgs))
#         print("  test     | {:5d} | {:8d} ".format(num_test_pids, num_test_imgs))
#         print("  ----------------------------------------")
