import os

class Logger():
    def __init__(self, savemodel_path, accelerator=None):
        self.accelerator = accelerator
        if self.accelerator is None or self.accelerator.is_local_main_process:    
            file_path = os.path.join(savemodel_path, "log.txt")
            self.file_= open(file_path, "a+")
            self.counter = 0
    def __del__(self):
        if self.accelerator is None or self.accelerator.is_local_main_process:
            self.file_.close()
    def log_and_print(self, contents):
        if self.accelerator is None or self.accelerator.is_local_main_process:
            self.counter+=1
            self.file_.write(str(contents) + "\n")
            print(contents)  
            if self.counter == 5:
                self.file_.flush()
                self.counter = 0