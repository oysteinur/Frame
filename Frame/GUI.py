from tkinter import *
#from Frame import * 
"""
def main():
    root = Tk()
    window1 = main_window(master)
    return None
"""

class main_window:
    n = 0
    
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        
        self.printButton = Button(frame, text = 'print message', command = self.printMessage)
        self.printButton.pack(side = LEFT)
        
        self.quitButton = Button(frame, text = 'Quit', command = frame.quit)
        self.quitButton.pack(side= LEFT)
        
    

    def printMessage(self):
        print('It works')
        
    


root = Tk()
window = main_window(root)
root.mainloop()
        
        
        
"""
master = tk.Tk()

label = tk.Label(master, text = 'Ramme')

label.pack()

master.mainloop()

master.configure(bg = 'light grey')

Header = tk.Label(text = 'Test')
Header.pack()
master.mainloop()
"""