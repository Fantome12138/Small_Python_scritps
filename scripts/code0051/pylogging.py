import logging
import logging.handlers

class loggings(object):
    
    def __init__(self, loggername=None,):
        self.mylogging = logging.getLogger(loggername)
        self.mylogging.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s[line:%(lineno)d]- %(levelname)s:  %(funcName)s - %(message)s")
        rotatingHandler = logging.handlers.TimedRotatingFileHandler(filename='/home/test.log', 
                                                                    when='MIDNIGHT', interval=1, backupCount=5)
        rotatingHandler.setFormatter(formatter)
        self.mylogging.addHandler(rotatingHandler)
        
    def getlog(self, ):
        return self.mylogging
    