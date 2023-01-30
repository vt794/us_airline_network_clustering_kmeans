import wget
from docopt import docopt
import pandas as pd
import tempfile
import zipfile
from zipfile import ZipFile
import sys
import shutil
from shutil import *
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Download():

    def __init__(self, args):
        """
        Defines class variables from arguments passed by docopt. 

        Assigns int dtype to `year` and `quarter` mandatory params, and 
        assign str dtype to optional `save_dir` param. 
        """

        self.args = args

        self.year = int(self.args['--year'])
        self.quarter = int(self.args['--quarter'])
        self.save_dir = str(self.args['--save_dir'])

    def download_unzip(self):

        def progress_bar(current, total, width=100):
            """
            Defines wget helper progress bar.

            Args:
                current (int): current byte count
                total (int): estimated total byte count
                width (int, optional): bar width, defaults to 100.
            """

            progress_message = "%d%% [%d / %d] bytes" % (
                (current/total) * 100, current, total)
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        for file_type in ["coupon", "ticket"]:

            url = ("https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_" +
                   f"DB1B{str.capitalize(file_type)}_{self.year}_{self.quarter}.zip")

            filename = (f"Origin_and_Destination_Survey_" +
                        f"DB1B{str.capitalize(file_type)}_{self.year}_{self.quarter}")
            filename_zip = filename + ".zip"
            filename_csv = f"{file_type}s_{self.year}_{self.quarter}"

            print(
                f"\ndownloading DB1B_{str.capitalize(file_type)}_{self.year}_{self.quarter}:")

            tempdir = tempfile.mkdtemp()
            wget.download(url, tempdir, bar=progress_bar)

            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)

            with zipfile.ZipFile(tempdir + "/" + filename_zip, "r") as zip_ref:
                zip_ref.extractall(self.save_dir + "/raw")

            os.rename(self.save_dir + "raw/" + filename + ".csv",
                      self.save_dir + "raw/" + filename_csv + ".csv")
            os.rename(self.save_dir + "raw/" + "/readme.html",
                      self.save_dir + "raw/" + f"/readme_{file_type}s.html")

            shutil.rmtree(tempdir)

    def test_output(self):
        """
        Tests the output tables to ensure relevance to time period query.
        """

        print(f"\ntesting coupon and ticket data for time period validity")

        for file_type in ["coupon", "ticket"]:

            df = pd.read_csv(self.save_dir + "raw/" + f"{file_type}s_{self.year}_{self.quarter}" + ".csv",
                             usecols=["Year", "Quarter"])

            assert (df["Year"].to_numpy() == self.year).all()
            assert (df["Quarter"].to_numpy() == self.quarter).all()


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    download = Download(args)
    download.download_unzip()
    download.test_output()
