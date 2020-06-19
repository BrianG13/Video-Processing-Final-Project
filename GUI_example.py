from tkinter import Tk, Label, Button
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

"""
Example for a Graphical User Interface.
Using tkinter library.
"""


def is_image(file_name):
    ret_val = False
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        ret_val = True
    else:
        print("The file {} is not an image file! Please select a valid file".format(file_name))
    return ret_val


class ExampleGUI:
    def __init__(self, master_window):
        self.master_window = master_window
        master_window.title("Video Processing Lab GUI Example")
        self.filename = None
        self.crop_image_roi = None

        self.get_image_label = Label(master_window, text="Get Image:")
        self.get_image_label.pack()
        self.get_image_path_button = Button(master_window, text="Browser", command=self.select_image_via_browser)
        self.get_image_path_button.pack()

        self.show_image_label = Label(master_window, text="View Selected Image:")
        self.show_image_label.pack()
        self.show_image_button = Button(master_window, text="Show Image", command=self.show_image)
        self.show_image_button.pack()

        self.get_interest_region_label = Label(master_window, text="Select Interest Region:")
        self.get_interest_region_label.pack()
        self.get_interest_region_button = Button(master_window, text="Select",
                                                 command=self.select_interest_region)
        self.get_interest_region_button.pack()
        
        self.view_selected_region_label = Label(master_window, text="View Selected Region: ")
        self.view_selected_region_label.pack()
        self.view_selected_region_button = Button(master_window, text="View Selection",
                                                  command=self.view_selected_region)
        self.view_selected_region_button.pack()

    """
    Definition of functions that would be called in a case of a button click
    """
    def select_image_via_browser(self):
        self.filename = askopenfilename(initialdir=os.getcwd())
        print(self.filename)

    def show_image(self):
        if self.filename:
            if is_image(self.filename):
                image = mpimg.imread(self.filename)
                plt.imshow(image)
                plt.show()

    def select_interest_region(self):
        if self.filename:
            if is_image(self.filename):
                image = cv2.imread(self.filename)
                # Select ROI
                from_center = False
                roi = cv2.selectROI("Drag the rect from the top left to the bottom right corner of the forground object,"
                                    " then press ENTER.",
                                    image, from_center)
                # Crop image
                self.crop_image_roi = roi
                cv2.destroyAllWindows()
                cv2.waitKey(0)

    def view_selected_region(self):
        # Display cropped image
        if self.crop_image_roi:
            image = cv2.imread(self.filename)
            crop_image = image[int(self.crop_image_roi[1]):int(self.crop_image_roi[1]+self.crop_image_roi[3]),
                               int(self.crop_image_roi[0]):int(self.crop_image_roi[0]+self.crop_image_roi[2])]
            cv2.imshow("Selected Part", crop_image)
            cv2.waitKey(0)


if __name__ == "__main__":
    root = Tk()
    root.geometry("200x200")  # You want the size of the app to be 500x500
    example_gui = ExampleGUI(root)
    root.mainloop()

