from som import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog as fd
from matplotlib.figure import Figure
#import matplotlib.cm as cm

class SOM_UI(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("SOM")
		self.geometry("650x600")
		self.createWidgets()

	def createWidgets(self):
		# select file
		tk.Label(text="File name: ", font=('Comic Sans MS', 12)).grid(row=0, column=0)
		self.filename = tk.StringVar()
		print_filename = tk.Label(self, textvariable=self.filename, font=('Comic Sans MS', 12))
		print_filename.grid(row=0, column=1)
		tk.Button(self, text='Select File', font=('Comic Sans MS', 12), command=self.select_file).grid(row=0, column=2)

		# setting epoch
		tk.Label(text="Epoch: ", font=('Comic Sans MS', 12)).grid(row=1, column=0)
		self.epoch_box = tk.Spinbox(self, from_=0, to=200, font=('Comic Sans MS', 12))
		self.epoch_box.grid(row=1, column=1)

		tk.Button(master=self, text='Start', font=('Comic Sans MS', 12), command=self.draw_picutre).grid(row=0, column=5)
		tk.Button(master=self, text='Exit', font=('Comic Sans MS', 12), command=self._quit).grid(row=1, column=5)
		
		# figures
		self.figure_result = Figure(figsize=(5,5), dpi=100)
		self.result_plt = FigureCanvasTkAgg(self.figure_result, self)
		self.result_plt.get_tk_widget().grid(row=6, column=1, columnspan=3)
		self.result_ax = self.figure_result.add_subplot(111)
		self.result_ax.set_title('Result')

	def draw_picutre(self):
		self.result_ax.clear()

		epochs = int(self.epoch_box.get())
		weights, label_map = process(self.File, epochs)
		
		self.result_ax = self.figure_result.add_subplot(111)
		self.result_ax.set_title('Result')
		self.result_ax.set_xlabel("1's dimension")
		self.result_ax.set_ylabel("2's dimension")

		self.result_ax.scatter(weights[:, 0], weights[:, 1], s=10, c=label_map, cmap='coolwarm')
		self.result_plt.draw()
		"""
		cmap = cm.get_cmap('Pastel1', classes)
		self.image = self.result_ax.imshow(label_map, cmap=cmap)	
		#self.result_ax.figure.colorbar(self.image)
		self.cbar = self.result_ax.figure.colorbar(self.image)
		"""

	def select_file(self):
		filetypes = (('text files', '*.txt'), ('All files', '*.*'))
		self.File = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
		file = ""
		for i in range(len(self.File) - 1, 0, -1):
			if self.File[i] == '/':
				file = self.File[i+1:]
				break
		self.filename.set(file)

	def _quit(self):
		self.quit()
		self.destroy()


if __name__ == "__main__":
	app = SOM_UI()
	app.mainloop()