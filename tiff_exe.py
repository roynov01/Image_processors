import tkinter as tk
from tkinter import filedialog
import pyperclip
import numpy as np
import tifffile as tiff
import tkinter.font as fnt
from tkinter.scrolledtext import ScrolledText
import re
from tkinter import ttk
import os

# DEFAULT_PATH = r"D:\python projects\small projects"
DEFAULT_PATH = os.getcwd()
LABEL_MAX_LEN = 13
COLORS = {1: "rgb", 2: "rgb", 3: "rgb", 4: "cmyk", 5: "ycbcr"}

class TiffEdit:
    """
    can split a tiff file into color channels
    and / or 
    merge few tif files into one multichannel + creates a max projection of them
    """
    def __init__(self, root):
        self.root = root
        self.root.title('smFISH dots enhancement')
        self.root.geometry('1050x560')
        self.root.iconbitmap('tiff.ico')
        self.root.configure(background='light blue')

        self.directory = None
        self.names = None
        self.files, self.files_ordered = None, None
        self.messages = []

        self.button_choose_file = tk.Button(self.root, text='Choose file', fg='black', bg='tomato',
                                           width=36, height=1, activebackground='blue', relief='raised',
                                           command=self.choose_tiff_file, state='normal', font=fnt.Font(size=20))
        self.button_names = tk.Button(self.root, text='OK', fg='black', bg='tomato',
                                           width=17, height=2, activebackground='blue', relief='raised',
                                           command=self.get_channel_names, state='disabled', font=fnt.Font(size=20))
        self.entry_names = ScrolledText(self.root, takefocus=1, cursor='xterm', width=30, height=5, bg='white', bd=5)
        self.button_matlab = tk.Button(self.root, text='Copy matlab command', fg='black', bg='tomato',
                                            width=17, height=1, activebackground='blue', relief='raised',
                                            command=self.matlab, state='normal', font=fnt.Font(size=20))
        self.button_dir = tk.Button(self.root, text='Copy directory', fg='black', bg='tomato',
                                            width=17, height=1, activebackground='blue', relief='raised',
                                            command=self.copy_dir, state='disabled', font=fnt.Font(size=20))
        self.button_files_merge = tk.Button(self.root, text='Choose files to merge', fg='black', bg='tomato',
                                            width=36, height=1, activebackground='blue', relief='raised',
                                            command=self.merge_files, state='normal', font=fnt.Font(size=20))
        self.button_merge = tk.Button(self.root, text='Merge', fg='black', bg='tomato',
                                      width=64, height=1, activebackground='blue', relief='raised',
                                      command=self.merge, state='normal', font=fnt.Font(size=20))
        self.button_reset = tk.Button(self.root, text='Reset', fg='white', bg='black',
                                      width=64, height=1, activebackground='red', relief='raised',
                                      command=self.reset, state='normal', font=fnt.Font(size=20))
        self.label_updates = tk.Label(self.root, text='', bg='PaleGreen1', justify='left', width=60, height=28,
                                      relief='groove', anchor='w')

        self.create_boxes()

        self.button_choose_file.grid(column=0, row=0, columnspan=2, padx=10, pady=3)
        self.button_names.grid(column=1, row=1, columnspan=1, padx=10, pady=3)
        self.entry_names.grid(column=0, row=1, padx=10, pady=3)
        self.button_matlab.grid(column=0, row=2, columnspan=1, padx=10, pady=3)
        self.button_dir.grid(column=1, row=2, columnspan=1, padx=10, pady=3)
        self.button_files_merge.grid(column=0, row=3, columnspan=2, padx=10, pady=3)
        self.label_updates.grid(column=3, row=0, columnspan=2, rowspan=9, padx=10, pady=3)
        self.button_merge.grid(column=0, row=9, columnspan=4, padx=10, pady=3)
        self.button_reset.grid(column=0, row=10, columnspan=4, padx=10, pady=3)

        self.choose_tiff_file()

    def box_call(self, event):
        """"detects changes in the comboboxes"""
        if not self.files:
            return
        for i in range(len(self.files)):
            self.files_ordered[i] = self.files.get(self.boxes[i].get())
        print(self.files_ordered)

    def create_boxes(self):
        """"creates comboboxes"""
        self.boxes = []
        for i in range(5):
            box = ttk.Combobox(self.root, width=43, height=1, textvariable=tk.StringVar(), state="disabled")
            box.bind('<<ComboboxSelected>>', self.box_call)
            box.grid(column=1, row=4+i, padx=10, pady=3)
            self.boxes.append(box)
            label = tk.Label(self.root, text=f'channel {i+1}', bg='light blue', width=30, height=1, anchor='e')
            label.grid(column=0, row=4+i, padx=10, pady=3)

    def choose_tiff_file(self):
        """"opens dialog to open a tif file"""
        self.filename = filedialog.askopenfilename(initialdir=DEFAULT_PATH, filetypes=[("*.tif", "TIF file")])
        if not self.filename:
            return
        self.button_names["state"] = "normal"
        self.button_dir["state"] = "normal"
        self.button_choose_file['bg'] = "violet"
        self.button_files_merge["bg"] = "tomato"
        self.button_dir["bg"] = "tomato"
        self.button_matlab["bg"] = "tomato"
        new = True if self.messages else False
        self.display(f"[OPENED] {self.filename}", new_line=new)
        self.tiff = tiff.imread(self.filename)
        self.tiff_array = np.array(self.tiff)
        directory = self.filename.split('/')
        self.directory = '/'.join(directory[:-1])

    def get_channel_names(self):
        """"gets names of channels, separated by comma or space or newline"""
        if not self.directory:
            return
        inputs = self.entry_names.get("1.0", tk.END)
        if inputs != '\n':  # if empty entry
            names = re.split(' |; |;|,|, |\n|\t', inputs)[:-1]
            names = [x for x in names if x != '']
            print("[NAMES CHOSEN] ", len(names), ": ", names)
            if len(names) == self.tiff_array.shape[1] and len(names) == len(set(names)):
                self.button_names["bg"] = "violet"
                self.names = names
                self.split_channels()
            else:
                self.button_names["bg"] = "tomato"
                self.display(f"Error - number of channel names is invalid, should be {self.tiff_array.shape[1]}")

    def split_channels(self):
        """"splits the channels and saves the images in .tif format"""
        for channel in range(self.tiff_array.shape[1]):
            cur_channel = self.tiff_array[:, channel, :, :]
            # tiff.imsave(f'{self.filename[:-4]}_{self.names[channel]}.tif', cur_channel, planarconfig='separate', dtype=np.uint16, bigtiff=True, metadata={'axes': 'ZYX'})
            tiff.imsave(f'{self.filename[:-4]}_{self.names[channel]}.tif', cur_channel, dtype=np.uint16, bigtiff=True, metadata={'axes': 'ZYX'})
        self.display("[FILES SAVED]")
        
    def matlab(self):
        """"copies matlab command for log_output"""
        self.button_matlab["bg"] = "violet"
        self.button_dir["bg"] = "tomato"
        command = "log_output_script_ami()"
        self.copy_command(command, "Matlab command copied, make sure to add X:/Amichay/resources to pat")

    def copy_dir(self):
        """"copies the directory of the tif file chosen"""
        if self.directory:
            self.button_dir["bg"] = "violet"
            self.button_matlab["bg"] = "tomato"
            self.copy_command(self.directory, "[DIRECTORY COPIED]")

    def copy_command(self, copy, message):
        pyperclip.copy(copy)
        self.display(message)

    def display(self, message, new_line=True):
        """
        displays updates at the label_updates and prints them
        :param message: str to be displayed and printed
        :param new_line: bool. add '\n' before the string
        """
        if len(self.messages) > LABEL_MAX_LEN:
            del self.messages[0]
        if new_line:
            message = '\n' + message
        print(message)
        if self.messages and message == self.messages[-1]:  # prevent duplicate messages
            return
        self.messages.append(message)
        self.label_updates['text'] = '\n'.join(self.messages)

    def reset(self):
        self.__init__(self.root)

    def merge_files(self):
        """"opens the files to merge and puts them into the comboboxes"""
        files = filedialog.askopenfilenames(initialdir= self.directory if self.directory else DEFAULT_PATH, title="Choose files to merge")
        filenames = []
        if not files:
            return
        if len(files) > 5:
            self.display("Number of channels cant exceed 5")
            return
        
        directory = files[0].split('/')
        self.directory = '/'.join(directory[:-1])
        self.create_boxes()
        self.button_files_merge["bg"] = "violet"
        self.button_merge["bg"] = "tomato"
        for file in files:
            filename = file.split('/')
            filename = '/'.join(filename[-1:])
            filenames.append(filename)
        for box in self.boxes[:len(files)]:
            box['values'] = filenames
            box["state"] = "readonly"
        self.files = {filenames[i]: files[i] for i in range(len(files))}
        self.files_ordered = [f for f in files]

    def merge(self):
        """"merges the chaneels and saves as one .tif file and one max_projection .tif file"""
        if not self.files or not all(self.files_ordered):
            self.display("Choose files to merge", True if self.messages else False)
            return
        lists = []
        lists_max = []
        for i in range(len(self.files_ordered)):
            img = tiff.imread(self.files_ordered[i])
            imarray = np.array(img)
            lists.append(imarray)
            lists_max.append(np.max(imarray, axis=0))
        lists = np.array(lists)
        lists_max = np.array(lists_max)
        print(f'shapes: {lists.shape},0:{lists_max.shape}')
        final_arr = np.transpose(lists, (1, 0, 2, 3))
        filename = filedialog.asksaveasfilename(initialdir=self.directory, title="Save merged TIFF",
                                                confirmoverwrite=True, filetypes=[("*.tif", "TIF file")])
        if not filename:
            return
        if not filename.endswith('.tif'):
            filename += '.tif'
        filename_max = filename[:-4]+"_max.tif"
        tiff.imsave(filename, final_arr, planarconfig='separate', dtype=np.uint16, bigtiff=True, metadata={'axes': 'ZCYX'})
        tiff.imsave(filename_max, lists_max, planarconfig='separate', dtype=np.uint16, bigtiff=True, metadata={'axes': 'CYX'})
        self.button_merge["bg"] = "violet"
        self.display("[MERGED FILE SAVED]")


if __name__ == "__main__":
    window = tk.Tk()
    window.update_idletasks()
    TiffEdit(window)
    window.mainloop()
