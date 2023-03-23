import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import bigfish.stack as stack
import bigfish.plot as plot

C = 3 # 3 if image is ZXYC in Diego or Dora microscope, 1 if ZCXY in boots microscope


def ask_files():
    root = tk.Tk()
    file_list = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes=[("*.tif", "TIF file")])
    root.destroy()
    if not file_list:
        raise ValueError("No files have been chosen")
    return file_list


def load_files():
    files = ask_files()
    images = [Image(file) for file in files]
    return images


def configure_z(images):
    path = images[0].directory_output + '/z_configuration.csv'
    file = ''
    for i, img in enumerate(images):
        file += f'{img.image_name},1,{img.shape[0]}\n'
    while True:
        try:
            with open(path, 'w') as f:
                f.write(file)
            break
        except PermissionError:
            input("Close the Excel file first, then press enter")
    os.system(f'start excel.exe {path}')
    while True:
        user_input = input('change Z stacks (and save), then press enter ')
        if user_input == '':
            break
    z1, z2 = read_z_file(path, len(images))
    crop_images(images, z1, z2)


def read_z_file(file, img_num):
    z1, z2 = [], []
    with open(file, "r") as f:
        for line in f:
            try:
                val1 = int(line.split(sep=",")[1])
                val2 = int(line.split(sep=",")[2].strip('\n'))
            except ValueError:
                raise ValueError('numbers in CSV are not integers')
            z1.append(val1)
            z2.append(val2)
    if len(z1) == len(z2) and len(z1) == img_num:
        return z1, z2
    raise ValueError('Excel file is not the same length as number of images')


def crop_images(images, z1, z2):
    for i, image in enumerate(images):
        image.crop(z1[i], z2[i])
    print('[IMAGES CROPPED]')


def configure_channels(channels_num, directory):
    path = directory + '/channel_configuration.csv'
    file = 'channel number,log_filter,DAPI,analysis\n'
    while True:
        try:
            with open(path, 'w') as f:
                for i in range(channels_num):
                    file += f'{i},{0 if i == channels_num-1 else 1},{1 if i == channels_num-1 else 0},{0 if i == channels_num-1 else 1}\n'
                f.write(file)
            break
        except PermissionError:
            input("Close the Excel file first, then press enter")
    os.system(f'start excel.exe {path}')
    while True:
        user_input = input('change configuration and save (0,1 for True False), then press enter ')
        if user_input == '':
            break
    which_log, which_dapi, which_rna = read_configuration(path, channels_num)
    return which_log, which_dapi, which_rna


def read_configuration(file, channels_num):
    which_log,which_dapi, which_rna = [False]*channels_num, [False]*channels_num, [False]*channels_num
    with open(file, "r") as f:
        for i, line in enumerate(f):
            values = line.split(sep=",")
            if i == 0:
                continue
            if len(values) < 4:
                break
            try:
                which_log[i-1] = bool(int(values[1]))
                which_dapi[i-1] = bool(int(values[2]))
                which_rna[i-1] = bool(int(values[3]))
            except ValueError:
                raise ValueError('numbers in CSV are not integers')
    if len(which_log) == len(which_dapi) and len(which_log) == channels_num and len(which_rna) == len(which_log):
        return which_log, which_dapi, which_rna
    raise ValueError('Excel file is not the same length as number of images')


class Image:
    def __init__(self, path, original_name=None):
        self.path = path
        path_list = path.split(sep="/")
        self.original_name = original_name
        self.format = path_list[-1][-4:]
        self.image_name = path_list[-1][:-4]
        print(f'[CREATING] {self.image_name}')
        self.directory_input = "/".join(path_list[:-1])
        self.image = stack.read_image(path)
        self.dims = len(self.image.shape)
        self.shape = self.image.shape
        self.channel_num = self.shape[C] if len(self.shape) == 4 else 1
        self.channels = []
        self.mip = None
        self.log = None
        if self.directory_input.endswith("output"):
            self.directory_output = self.directory_input
        else:
            self.directory_output = self.directory_input + '/output'
            try:
                os.mkdir(self.directory_input + '/output')
            except FileExistsError:
                pass


    def max_project(self, log=True):
        if self.dims != 3:
            print("warning: max projection require 3 dimensions")
            return
        print(f'[MAX PROJECTING] {self.image_name}')
        if log and self.log:
            to_mip = self.log.image
            name = self.log.image_name + "_MAX" + self.format
        else:
            to_mip = self.image
            name = self.image_name + "_MAX" + self.format
        mip = stack.maximum_projection(to_mip)
        path = self.directory_output + '/' + name
        stack.save_image(mip, path, "tif")
        self.mip = Image(path, original_name=self.get_name())
        self.mip.save()
        return self.mip

    def crop(self, z1, z2):
        print(f'[CROP] {self.image_name}')
        self.image = self.image[z1 - 1:z2 - 1, :, :]
        self.shape = self.image.shape

    def save(self):
        stack.save_image(self.image, self.path, "tif")

    def split_channels(self):
        if self.dims < 3:
            print("cant perform LOG_FILTER on 2D image")
            return
        print(f'[SPLIT] {self.image_name}')
        self.channels = []
        for chan in range(self.image.shape[C] if self.dims == 4 else self.image.shape[0]):
            if C == 1:
                cur_channel = self.image[:, chan, :, :]
            elif C == 3:
                cur_channel = self.image[:, :, :, chan]
            path = f'{self.directory_output}/{self.image_name}_{chan + 1}.tif'
            stack.save_image(cur_channel, path, "tif")
            image = Image(path, original_name=self.get_name())
            image.save()
            self.channels.append(image)
        return self.channels

    def log_filter(self, sigma=3):
        if self.dims < 3:
            print("warning: cant perform LOG_FILTER on 2D image")
            return
        log = stack.log_filter(self.image, sigma=sigma)
        print(f'[LOG_FILTER] {self.image_name}')
        path = self.directory_output+ '/' + self.image_name + "_LOG.tif"
        stack.save_image(log, path, "tif")
        self.log = Image(path, original_name=self.get_name())
        self.log.save()
        return self.log

    def __str__(self):
        return f'* image {self.image_name} of shape {self.shape}\n' \
               f'* path of image: {self.path}'

    def get_name(self):
        return self.original_name if self.original_name else self.image_name

    def show(self, contrast=False):
        if self.dims != 2:
            print("warning: plotting require 2 dimensions")
            return
        plot.plot_images(self.image, rescale=True, titles=self.image_name)



def merge_channels(img_list):
    import tifffile as tiff
    imarray = np.array([img.image for img in img_list])
    imarray_max = np.array([np.max(img.image, axis=0) for img in img_list])
    directory = img_list[0].directory_output
    name = img_list[0].get_name()
    print(f'[MERGED] {name}, shape = {imarray.shape}')
    path = os.path.join(directory, name + "_MERGED.tif")
    path_max = os.path.join(directory, name + "_MERGED_MAX.tif")
    imarray = np.transpose(imarray, (1, 0, 2, 3))
    tiff.imsave(path, imarray, planarconfig='separate', dtype=np.uint16, bigtiff=True, metadata={'axes': 'ZCYX'})
    tiff.imsave(path_max, imarray_max, planarconfig='separate', dtype=np.uint16, bigtiff=True, metadata={'axes': 'CYX'})


os.chdir(r"C:\Users\royno\Desktop\py\bigfish\test\image_automation")

analysis = False
# if input("press y to include analysis: ") == "y":
#     analysis = True


images = load_files()
configure_z(images)
which_log, which_dapi, which_rna = configure_channels(images[0].channel_num, images[0].directory_output)
for i, image in enumerate(images):
    image.split_channels()
    
    images_to_merge = []
    name = image.image_name
    for j, chan in enumerate(image.channels):
        if which_log[j]:
            log = chan.log_filter(sigma=1)
            images_to_merge.append(log)
            chan.max_project(log=True)
        else:
            chan.max_project(log=False)
            images_to_merge.append(chan)
        if which_rna[j]:
            rna = chan
    merge_channels(images_to_merge)



# image = Image(r"X:\roy\pancreas\ADAR_KO\microscopy_images\20220901_acly_actb_tests\18A_GCG-cy5_ACLY-a594\output/18A_GCG-cy5_ACLY-a594_6_1.tif")
# for i in range(1):
#     log = image.log_filter(0.5)
#     log_mip = log.max_project()
#     log_mip.show()
    