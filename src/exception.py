import sys
from src.logger import logging

def get_error_message(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    frame = exc_tb.tb_frame.f_code.co_filename

    message = "The Error occurred at file {0}, at line no {1} and the error is {2}".format(frame,exc_tb.tb_lineno,str(error))

    return message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_message = get_error_message(error,error_detail)

    def __str__(self) -> str:
        return self.error_message
    
