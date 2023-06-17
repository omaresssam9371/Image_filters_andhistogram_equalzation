from math import sqrt
from cv2 import BORDER_DEFAULT, CV_16S, imshow, merge
from matplotlib import image
from skimage.color import rgb2gray
from skimage.io import imread
import PIL.Image
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from skimage import io
from skimage import color
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy.fft import fft, fftfreq, ifft
from scipy import ndimage
from scipy.fft import fft, ifft
from scipy import fftpack
from PIL import ImageTk, Image, ImageFilter
import cv2
import numpy as np
import math
import matplotlib.image as mpimg
from tkinter import *
import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from sympy import I
from PIL import Image

root = Tk()
style = ttk.Style(root)

# tell tcl where to find the awthemes packages
# root.tk.eval("""
# set base_theme_dir awthemes-10.4.0

# package ifneeded awthemes 10.4.0 \
#     [list source [file join $base_theme_dir awthemes.tcl]]
# package ifneeded awdark 7.12 \
#     [list source [file join $base_theme_dir awdark.tcl]]
# """)

# root.tk.call("package", "require", 'awdark')
# root.tk.call("package", "require", 'awlight')
# style.theme_use('awdark')
style.configure('my.TMenubutton', font=('Arial', 15))
style.configure('my.TButton', font=('Arial', 15))

fig = plt.figure(figsize=(6, 4), facecolor='#1c1b1c')
# fig2 = plt.figure(figsize=(4, 3))
fig3 = plt.figure(figsize=(6, 4), facecolor='#ffffff')
fig4 = plt.figure(figsize=(6, 4), facecolor='#ffffff')
fig5 = plt.figure(figsize=(6, 4), facecolor='#000000')
fig6 = plt.figure(figsize=(6, 4), facecolor='#ffffff')
fig7 = plt.figure(figsize=(6, 4), facecolor='#ffffff')
fig8 = plt.figure(figsize=(6, 4), facecolor='#ffffff')

ax1 = fig.add_subplot()

ax3 = fig3.add_subplot()
ax4 = fig4.add_subplot()
ax5 = fig5.add_subplot()
ax6 = fig6.add_subplot()
ax7 = fig7.add_subplot()
ax8 = fig8.add_subplot()

root.title("Tab Widget")
tabControl = ttk.Notebook(root)

filters_tab = ttk.Frame(tabControl)
filters_tab.grid(row=0, column=0, sticky="nsew")

histogram_tab = ttk.Frame(tabControl)
histogram_tab.grid(row=0, column=0, sticky="nsew")

tabControl.add(filters_tab, text='filters')
tabControl.add(histogram_tab, text='histogram')
tabControl.grid(row=0, column=0, columnspan=2, sticky="nsew")


image_frame = LabelFrame(filters_tab, width=200, height=200)
image_frame.grid(row=1, column=1)

filters_frame = LabelFrame(filters_tab, width=300, height=300)
filters_frame.grid(row=1, column=0)

histogram_equalization_frame = LabelFrame(histogram_tab, width=400, height=400)
histogram_equalization_frame.grid(row=1, column=1)

settings_frame = LabelFrame(histogram_tab, width=200, height=200)
settings_frame.grid(row=1, column=0)

ax1.axes.yaxis.set_visible(False)
ax1.axes.xaxis.set_visible(False)
ax1.set_title('original image',color='white')


ax3.axes.yaxis.set_visible(False)
ax3.axes.xaxis.set_visible(False)
ax3.set_title("image filterd in spatial",color='white')

ax4.axes.yaxis.set_visible(False)
ax4.axes.xaxis.set_visible(False)
ax3.set_title("image filterd in spatial",color='white')



ax5.axes.yaxis.set_visible(False)
ax5.axes.xaxis.set_visible(False)
ax5.set_title('original image',color='white')

ax6.axes.yaxis.set_visible(True)
ax6.axes.xaxis.set_visible(True)
ax6.set_title("original image histogram",color='white')
ax6.set_xlabel("Intensity",color='white')
ax6.set_ylabel("Count",color='white')
ax6.tick_params(axis='x', colors='white')
ax6.tick_params(axis='y', colors='white')

ax7.axes.yaxis.set_visible(False)
ax7.axes.xaxis.set_visible(False)
ax7.set_title("equalized image",color='white')


ax8.axes.yaxis.set_visible(True)
ax8.axes.xaxis.set_visible(True)
ax8.set_title('equalized image histogram',color='white')
ax8.set_xlabel("Intensity",color='white')
ax8.set_ylabel("Count",color='white')
ax8.tick_params(axis='x', colors='white')
ax8.tick_params(axis='y', colors='white')


def is_grey_scale(img_path):
    img = PIL.Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def browse():
    global gray_image
    global rgb_image
    global file_path
    global original_image
    file_path = filedialog.askopenfilename()
    original_image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_image)
    ax3.cla()
    ax4.cla()
    filterd_image_canvas.draw_idle()
    frequency_image_canvas.draw_idle()
    original_image_canvas.draw_idle()



def laplacian_filter():
    global rgb_image
    if(is_grey_scale(file_path) == True):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        laplacian_image = cv2.Laplacian(gaussian_image, cv2.CV_64F)
        ax3.imshow(laplacian_image, cmap='gray')
        filterd_image_canvas.draw_idle()
        fourierTransform(laplacian_image)
    else:
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        [h, s, v] = cv2.split(hsv)
        gaussian_image = cv2.GaussianBlur(v, (3, 3), 0)
        laplacian_image = cv2.Laplacian(
            v, cv2.CV_8U, borderType=BORDER_DEFAULT)
        merged_image = cv2.merge([h, s, laplacian_image])
        rgb_again = cv2.cvtColor(merged_image, cv2.COLOR_HSV2RGB)
        ax3.imshow(rgb_again)
        filterd_image_canvas.draw_idle()
        fourierTransform(laplacian_image)


def low_pass():
    global rgb_image
    global gray_image

    ksize = (10, 10)
  
# Using cv2.blur() method 
    image_spatial = cv2.blur(rgb_image, ksize) 
    
    # Displaying the image 
    
    if (is_grey_scale(file_path)==True):
        ax3.imshow(image_spatial,cmap='gray')
        frequency_image_canvas.draw_idle()
        gray_image=cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
        filtered_image=applying_low_pass(gray_image)
        ax4.imshow(filtered_image,cmap='gray')
        ax4.set_title('image filtered in fourier',color='white')
        ax3.set_title("image filterd in spatial",color='white')

        filterd_image_canvas.draw_idle()

    else :
        ax3.imshow(image_spatial)
        frequency_image_canvas.draw_idle()

        # return img2
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        [h,s,v]=cv2.split(hsv)
        filtered_image4=applying_low_pass(v)
        u8_filtered2=np.uint8(filtered_image4)
        merged_image=cv2.merge([h,s,u8_filtered2])
        rgb_again=cv2.cvtColor(merged_image,cv2.COLOR_HSV2RGB)
        ax4.imshow(rgb_again)
        ax3.set_title("image filterd in spatial",color='white')

        ax4.set_title('image filtered in fourier',color='white')
        filterd_image_canvas.draw_idle()

def applying_low_pass(image):
    h,w =image.shape[0:2]

    # for i in range(2000):    
    #     x = np.random.randint(0, h)
    #     y = np.random.randint(0, w)
    #     image[x,y] = 255

    img_dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(img_dft)  
    # h, w = img.shape
    h1,w1 = int(h/2), int(w/2)
    img2 = np.full((h, w),(1), np.uint8)*0.1
    k=45
    img2[(h1-int(k/2)):(h1+int(k/2)), (w1-int(k/2)):(w1+int(k/2))] = 1

    # dft_shift=dft_shift*img2*.9
    dft_shift = np.log(np.abs(img2)+1)

    idft_shift = np.fft.ifftshift(dft_shift)  
    ifimg = np.fft.ifft2(idft_shift)  
    ifimg = np.abs(ifimg)
    return ifimg


def high_Pass():
    global rgb_image
    global file_path
    if (is_grey_scale(file_path) == True):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        filtered_image_fourier = applying_high_pass(gray_image)
        kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0 ,-1.0, 0.0]])

                           
        filtered_image = cv2.filter2D(gray_image, -1, kernel)
        ax3.imshow(filtered_image,cmap='gray')
        filterd_image_canvas.draw_idle()
        ax4.imshow(filtered_image_fourier, cmap='gray')
        ax4.set_title('image filtered in fourier',color='white')
        ax3.set_title("image filterd in spatial",color='white')
        filterd_image_canvas.draw_idle()
        frequency_image_canvas.draw_idle()

    else:
        [h, s, v] = cv2.split(rgb_image)
        filtered_image_fourier = applying_high_pass(v)
        kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

        resulting_image = cv2.filter2D(v, -1, kernel)
        u8_filtered_image_fourier = np.uint8(filtered_image_fourier)
        u8_filtered_image_spatial = np.uint8(resulting_image)
        merged_image_fourier = cv2.merge([h, s, u8_filtered_image_fourier])
        merged_image_spatial=cv2.merge([h,s,u8_filtered_image_spatial])
        rgb_again_fourier = cv2.cvtColor(merged_image_fourier, cv2.COLOR_HSV2RGB)
        rgb_again_spatial = cv2.cvtColor(merged_image_spatial, cv2.COLOR_HSV2RGB)
        ax4.imshow(rgb_again_fourier)
        ax3.imshow(rgb_again_spatial)
        ax4.set_title('image filtered in fourier',color='white')
        ax3.set_title("image filterd in spatial",color='white')
        filterd_image_canvas.draw_idle()
        frequency_image_canvas.draw_idle()


def applying_high_pass(image):
    fimg = np.fft.fft2(image)
    fshift = np.fft.fftshift(fimg)
    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)
    fshift[crow-25:crow+25, ccol-25:ccol+25] = 0
    magnitude_spectrum = np.log(1+np.abs(fshift))
    ax4.imshow(magnitude_spectrum, cmap='gray')
    frequency_image_canvas.draw_idle()
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg



def median_filter():
    global rgb_image
    if(is_grey_scale(file_path) == True):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        filtered = cv2.medianBlur(gray_image, 9)
        ax3.imshow(filtered, cmap=plt.get_cmap('gray'))
        filterd_image_canvas.draw_idle()
        fourierTransform(filtered)
    else:
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        [h, s, v] = cv2.split(hsv)
        median_image = cv2.medianBlur(v, 9)
        merged_image = cv2.merge([h, s, median_image])
        rgb_again = cv2.cvtColor(merged_image, cv2.COLOR_HSV2RGB)
        ax3.imshow(rgb_again)
        filterd_image_canvas.draw_idle()
        fourierTransform(median_image)


def fourierTransform(filteredSignal):
    f = np.fft.fft2(filteredSignal)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum_log = np.log(magnitude_spectrum+1)
    ax4.imshow(magnitude_spectrum_log, cmap=plt.get_cmap('gray'))
    ax4.set_title('log spectrum',color='white')
    ax3.set_title("image filterd in spatial",color='white')

    frequency_image_canvas.draw_idle()


def browse_histogram():
    global original_image
    global gray_image_histogram
    file_path = filedialog.askopenfilename()
    original_image = cv2.imread(file_path)
    original_image = cv2.resize(original_image, (800, 600))
    gray_image_histogram = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image_histogram = cv2.convertScaleAbs(
        gray_image_histogram, alpha=1.10, beta=-20)
    ax5.imshow(gray_image_histogram, cmap='gray')
    ax6.cla()
    ax7.cla()
    ax8.cla()
    original_image_canvas_histogram.draw_idle()
    equalized_image_canvas.draw_idle()
    frequency_image_canvas.draw_idle()
    equalized_image_histogram_canvas.draw_idle()


def Histogram_of_Image(img):
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram


def plot_histogram():
    img = np.array(gray_image_histogram).flatten()
    histogram = Histogram_of_Image(img)
    x_axis = np.arange(256)
    ax6.set_xlabel("Intensity")
    ax6.set_ylabel("Count")
    ax6.bar(x_axis, histogram)
    original_image_histogram_canvas.draw_idle()


def cumulative_sum(histogram):
    cumulativesum = np.zeros(256, dtype=int)
    cumulativesum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumulativesum[i] = cumulativesum[i-1] + histogram[i]
    return cumulativesum


def make_mapping(cumulativesum):
    mapping = np.zeros(256, dtype=int)
    gray_levels = 256
    height, width = gray_image_histogram.shape
    for i in range(gray_levels):
        mapping[i] = max(
            0, round((gray_levels*cumulativesum[i])/(height*width))-1)
    return mapping


def apply_mapping(img, mapping):
    new_image = np.zeros(img.size, dtype=int)
    for i in range(img.size):
        new_image[i] = mapping[img[i]]
    return new_image


def show_image():
    img = np.array(gray_image_histogram).flatten()
    histogram = Histogram_of_Image(img)
    cumulativesum = cumulative_sum(histogram)
    mapping = make_mapping(cumulativesum)
    equalized_image = apply_mapping(img, mapping)
    height, width = gray_image_histogram.shape
    output_image = Image.fromarray(
        np.uint8(equalized_image.reshape((height, width))))
    x_axis = np.arange(256)
    ax8.set_xlabel("Intensity")
    ax8.set_ylabel("Count")
    ax8.bar(x_axis, Histogram_of_Image(equalized_image), color="orange")
    equalized_image_histogram_canvas.draw_idle()
    ax7.imshow(output_image, cmap='gray')
    equalized_image_canvas.draw_idle()


browse_button_histogram = ttk.Button(histogram_tab, text='browse', command=lambda: [
                                     browse_histogram(), plot_histogram()], style='my.TButton').grid(row=0, column=0, sticky='nw')
histogram_equalization_button = ttk.Button(settings_frame, text='apply histogram equalization',
                                           command=show_image, style='my.TButton').grid(row=0, column=0, sticky='nw')


browse_button = ttk.Button(filters_tab, text='browse', command=browse, style='my.TButton').grid(
    row=0, column=0, sticky='nw')
laplacian_button = ttk.Button(filters_frame, text='la placian filter',
                              command=laplacian_filter, style='my.TButton').grid(row=1, column=0, sticky='nsew')
lowpass_button = ttk.Button(filters_frame, text='lowpass filter',
                            command=low_pass, style='my.TButton').grid(row=2, column=0, sticky='nsew')
highpass_button = ttk.Button(filters_frame, text='highpass filter',
                             command=high_Pass, style='my.TButton').grid(row=3, column=0, sticky='nsew')
Median_button = ttk.Button(filters_frame, text='Median filter',
                           command=median_filter, style='my.TButton').grid(row=4, column=0, sticky='nsew')

original_image_canvas = FigureCanvasTkAgg(fig, master=image_frame)
original_image_canvas.draw()
original_image_canvas.get_tk_widget().grid(
    row=0, column=0, sticky=" nsew", columnspan=2)

filterd_image_canvas = FigureCanvasTkAgg(fig3, master=image_frame)
filterd_image_canvas.draw()
filterd_image_canvas.get_tk_widget().grid(row=1, column=0)
frequency_image_canvas = FigureCanvasTkAgg(fig4, master=image_frame)
frequency_image_canvas.draw()
frequency_image_canvas.get_tk_widget().grid(row=1, column=1)

original_image_canvas_histogram = FigureCanvasTkAgg(
    fig5, master=histogram_equalization_frame)
original_image_canvas_histogram.draw()
original_image_canvas_histogram.get_tk_widget().grid(row=0, column=0)
original_image_histogram_canvas = FigureCanvasTkAgg(
    fig6, master=histogram_equalization_frame)
original_image_histogram_canvas.draw()
original_image_histogram_canvas.get_tk_widget().grid(row=0, column=1)
equalized_image_canvas = FigureCanvasTkAgg(
    fig7, master=histogram_equalization_frame)
equalized_image_canvas.draw()
equalized_image_canvas.get_tk_widget().grid(row=1, column=0)
equalized_image_histogram_canvas = FigureCanvasTkAgg(
    fig8, master=histogram_equalization_frame)
equalized_image_histogram_canvas.draw()
equalized_image_histogram_canvas.get_tk_widget().grid(row=1, column=1)


####### grid configuratio filters tab #####
Grid.rowconfigure(filters_tab, 0, weight=1)
Grid.rowconfigure(filters_tab, 1, weight=1)
Grid.columnconfigure(filters_tab, 0, weight=1)
Grid.columnconfigure(filters_tab, 1, weight=1)

Grid.rowconfigure(image_frame, 0, weight=1)
Grid.columnconfigure(image_frame, 0, weight=1)

Grid.rowconfigure(image_frame, 1, weight=1)
Grid.columnconfigure(image_frame, 1, weight=1)

Grid.rowconfigure(filters_frame, 0, weight=1)
Grid.columnconfigure(filters_frame, 0, weight=1)

Grid.rowconfigure(filters_frame, 2, weight=1)
Grid.rowconfigure(filters_frame, 3, weight=1)
Grid.rowconfigure(filters_frame, 4, weight=1)

####### grid configuration histogram tab#####
Grid.rowconfigure(histogram_tab, 0, weight=1)
Grid.rowconfigure(histogram_tab, 1, weight=1)
Grid.columnconfigure(histogram_tab, 0, weight=1)
Grid.columnconfigure(histogram_tab, 1, weight=1)

Grid.rowconfigure(histogram_equalization_frame, 0, weight=1)
Grid.columnconfigure(histogram_equalization_frame, 0, weight=1)

Grid.rowconfigure(histogram_equalization_frame, 1, weight=1)
Grid.columnconfigure(histogram_equalization_frame, 1, weight=1)

Grid.rowconfigure(settings_frame, 0, weight=1)
Grid.columnconfigure(settings_frame, 0, weight=1)

root.mainloop()  # Start the GUI
