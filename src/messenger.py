import logging
import slackweb

class Info(object):
    def __init__(self, info_prefix='', slack_url = None):
        
        self.info_prefix = info_prefix
        self.slack = None
        if slack_url is not None:
            self.slack = slackweb.Slack(url = slack_url)
            self.slack.notify(text = "="*80)
        
    def print_msg(self, msg):
        text = self.info_prefix + ' ' + msg
        
        print(text)
        logging.info(text)
        if self.slack is not None:
            self.slack.notify(text = text)
            
        
        