"""
Downloads DB1B data based on time period of interest specified by the user, and 
applies transformations into ticket-indexed data.

Usage:
    src/data_pipeline.py [--year=<year> --quarter=<quarter>] [options]
    src/data_pipeline.py (-h | --help)
    src/data_pipeline.py

Options:
    --year=<year>                        Year of ticketing data extracted (in 4-digit numeric format).
    --quarter=<quarter>                  Quarter of ticketing data extracted (in 1-digit numeric format).
    --save_dir=<save_dir>                Path to data directory [default: ./data/].
    --help                               Prints this help prompt.
    --version                            Prints version.
    
    
"""


from download import Download
from preprocessing import Preprocess

from docopt import docopt


class DataPipeline():

    def __init__(self, args):
        """__init__ Init for pipeline

        Args:
            args (docopt): Arguments passed through docopt
        """

        self.args = args

    def extract(self):
        """
        Extracts DB1B coupon and ticket data for the quarter and year specified.

        Downloads data in zipped csv format to a temporary directory, and unzips 
        the csv to a raw data directory. 
        """

        extract = Download(args)
        extract.download_unzip()
        extract.test_output()

    def transform(self):
        """
        Preprocesses DB1B coupon and ticket data 
        """

        preprocess = Preprocess(args)
        preprocess.preprocess()
        preprocess.save_tickets_pp()


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    data_pipeline = DataPipeline(args)
    data_pipeline.extract()
    data_pipeline.transform()
